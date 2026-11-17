#include <xpool/devkit/ffn_routing_observer.hpp>

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda/std/span>
#include <cuda_runtime.h>

#include <xpool/arena.hpp>
#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/timeline/attachments.hpp>
#include <xpool/devkit/timeline/collection.cuh>
#include <xpool/devkit/timeline/routingevent.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>
#include <xpool/utils/time.cuh>
#include <xpool/utils/trace.cuh>

namespace xpool::devkit::ffn_routing_observer {

namespace {

struct RoutingState {
  timeline::DeviceAttemptState attempts{};
  std::uint64_t oversized = 0;
  std::uint64_t full = 0;
};
struct RecordHeader {
  std::uint64_t generation;
  std::int32_t pe;
  bool oversized;
  std::uint64_t attachment_sequence;
  std::uint64_t raw_time_ns;
  xpool::fabric::InvocationKey key;
  std::size_t layer_ordinal;
  std::size_t executor_lane_index;
  std::uint64_t executor_lease_sequence;
  std::uint64_t rows;
  std::uint64_t effective_topk;
};

struct BufferLayout {
  std::size_t record_capacity;
  std::size_t element_capacity;
  std::size_t state_offset;
  std::size_t headers_offset;
  std::size_t ids_offset;
  std::size_t weights_offset;
  std::size_t total_bytes;

  static BufferLayout create(std::size_t record_capacity, std::size_t element_capacity) {
    TORCH_CHECK(element_capacity != 0, "routing observer element capacity must be positive");
    const auto total_elements = xpool::utils::checked::prod(record_capacity, element_capacity);
    const auto plan = xpool::utils::layout::LayoutPlan<4>{
        std::array{
            xpool::utils::layout::LayoutRegionSpec::object<RoutingState>("state"),
            xpool::utils::layout::LayoutRegionSpec::array<RecordHeader>("headers", record_capacity),
            xpool::utils::layout::LayoutRegionSpec::array<std::int32_t>("ids", total_elements),
            xpool::utils::layout::LayoutRegionSpec::array<float>("weights", total_elements),
        },
        xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    const auto &state = plan[index++];
    const auto &headers = plan[index++];
    const auto &ids = plan[index++];
    const auto &weights = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool FFN Routing Observer region plan is incomplete");
    return BufferLayout{
        .record_capacity = record_capacity,
        .element_capacity = element_capacity,
        .state_offset = state.offset,
        .headers_offset = headers.offset,
        .ids_offset = ids.offset,
        .weights_offset = weights.offset,
        .total_bytes = plan.total_bytes,
    };
  }
};

class BufferView {
public:
  XPOOL_HOST_DEVICE_FN BufferView(std::uint8_t *base, BufferLayout layout) : base_(base), layout_(layout) {}

  XPOOL_HOST_DEVICE_FN std::uint8_t *base() const { return base_; }
  XPOOL_HOST_DEVICE_FN const BufferLayout &layout() const { return layout_; }

  XPOOL_HOST_DEVICE_FN RoutingState &state() const {
    return *reinterpret_cast<RoutingState *>(base_ + layout_.state_offset);
  }

  XPOOL_HOST_DEVICE_FN std::int32_t *ids(std::size_t index) const {
    return reinterpret_cast<std::int32_t *>(base_ + layout_.ids_offset) + index * layout_.element_capacity;
  }

  XPOOL_HOST_DEVICE_FN float *weights(std::size_t index) const {
    return reinterpret_cast<float *>(base_ + layout_.weights_offset) + index * layout_.element_capacity;
  }

private:
  std::uint8_t *base_;
  BufferLayout layout_;
};

struct Storage {
  std::uint8_t *buffer = nullptr;
  BufferLayout layout{};
  timeline::RoutingSource source{};
};

static_assert(std::is_trivially_copyable_v<Storage>);

XPOOL_DEVICE_CONST Storage storage{};

std::mutex state_mutex;
std::uint8_t *buffer = nullptr;
BufferLayout layout{};
bool installed = false;
bool collected = false;
timeline::MemoryBudget::Reservation device_charge;
struct RoutingCopy {
  timeline::MemoryBudget::Reservation charge;
  RoutingState state{};
  std::vector<RecordHeader> headers;
  std::vector<timeline::RoutingAttachment> records;
};
std::shared_ptr<RoutingCopy> pending_copy;
bool copy_uncertain = false;
bool counts_recorded = false;
std::size_t copy_bytes(std::size_t records, std::size_t elements) {
  // Four payload-sized coexistence allowances cover CPU tensors, Safetensors
  // serialization and Python/Native byte handoff. Pointer-backed flattening
  // does not introduce a second tensor copy in the pinned Safetensors version.
  constexpr auto element_workspace_bytes = std::size_t{32};
  constexpr auto record_workspace_bytes = std::size_t{4096};
  constexpr auto fixed_workspace_bytes = std::size_t{65536};
  return xpool::utils::checked::sum(
      fixed_workspace_bytes,
      xpool::utils::checked::prod(
          records,
          xpool::utils::checked::sum(sizeof(timeline::RoutingAttachment), sizeof(RecordHeader), record_workspace_bytes,
                                     xpool::utils::checked::prod(elements, element_workspace_bytes))));
}

void copy_to_host(void *destination, const void *source, std::size_t bytes) {
  copy_uncertain = true;
  const auto status = cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    timeline::fail_routing_transfer("Routing Device-to-Host copy failed", cudaGetErrorString(status));
  copy_uncertain = false;
}

} // namespace

std::size_t element_capacity(const xpool::fabric::ArenaProjection &projection, std::size_t ffnagent_index) {
  auto element_capacity = std::size_t{0};
  for (const auto &instance : projection.instances) {
    const auto payload_row_capacity =
        std::max(instance.decode_payload_row_capacity, instance.prefill_payload_row_capacity);
    for (const auto &layer : instance.layers) {
      if (layer.kind == xpool::ffn::LayerKind::Moe && !layer.ffnagent_indices.empty() &&
          layer.ffnagent_indices.front() == ffnagent_index) {
        element_capacity =
            std::max(element_capacity, xpool::utils::checked::prod(payload_row_capacity, layer.effective_topk));
      }
    }
  }
  return element_capacity;
}

MemoryDemand memory_demand(const xpool::fabric::ArenaProjection &projection, std::size_t ffnagent_index) {
  if (ffnagent_index >= projection.ffnagent_count)
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, 0,
                                    "Routing sizing requires an actual FfnAgent index");
  const auto elements = element_capacity(projection, ffnagent_index);
  const auto &options = xpool::debug::options().ffn_routing_observer;
  if (!options.enable || elements == 0)
    return {};
  const auto host = copy_bytes(options.record_capacity, elements);
  return {allocation_bytes(elements), host};
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  TORCH_CHECK(!installed && buffer == nullptr && !pending_copy,
              "xpool Routing Observer still owns an installed or retained copy");
  counts_recorded = false;
  collected = false;
  const auto element_capacity =
      xpool::devkit::ffn_routing_observer::element_capacity(context.fabric_projection, context.ffnagent_index);
  if (element_capacity == 0) {
    installed = true;
    collected = true;
    return;
  }
  device_charge = timeline::admit_routing_device(allocation_bytes(element_capacity));
  if (element_capacity != 0) {
    layout = BufferLayout::create(xpool::debug::options().ffn_routing_observer.record_capacity, element_capacity);
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), layout.total_bytes));
    C10_CUDA_CHECK(cudaMemset(buffer, 0, layout.total_bytes));
  }
  const auto installed_storage = Storage{.buffer = buffer, .layout = layout, .source = timeline::routing_source()};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &installed_storage, sizeof(installed_storage)));
  installed = true;
  collected = false;
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable ||
      context.kind != xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataPublished ||
      storage.buffer == nullptr) {
    return;
  }

  const auto &execution = *context.execution;
  const auto &layer = context.arena.layer_entry(execution.key.instance_index, execution.layer_ordinal);

  const auto group = cooperative_groups::this_thread_block();
  const auto view = BufferView{storage.buffer, storage.layout};
  const auto metadata = context.arena.routing_metadata(context.executor_lane_index, context.payload_row_capacity);
  const auto live_element_count = execution.payload_rows * layer.effective_topk;
  XPOOL_DEVICE_SHARED std::uint64_t reserved_sequence;
  // One thread reserves and initializes the header before the CTA copies the
  // corresponding routing payload into the same one-based record slot.
  if (group.thread_rank() == 0) {
    const auto headers = cuda::std::span{reinterpret_cast<RecordHeader *>(view.base() + view.layout().headers_offset),
                                         view.layout().record_capacity};
    const auto sequence = timeline::detail::next_device_sequence(view.state().attempts);
    auto *entry = sequence > 0 && sequence <= view.layout().record_capacity ? &headers[sequence - 1] : nullptr;
    if (sequence == 0)
      timeline::detail::source(storage.source.events).stop_identity_exhausted();
    const auto raw_time = xpool::utils::time::now();
    const auto oversized =
        layer.effective_topk == 0 || execution.payload_rows > storage.layout.element_capacity / layer.effective_topk;
    if (sequence && oversized)
      cuda::atomic_ref{view.state().oversized}.fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed);
    if (sequence && !entry && !oversized)
      cuda::atomic_ref{view.state().full}.fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed);
    const auto attachment_state = oversized ? timeline::RoutingAttachmentState::PayloadOversize
                                  : entry   ? timeline::RoutingAttachmentState::Reserved
                                            : timeline::RoutingAttachmentState::CapacityExhausted;
    const auto event = timeline::make_routing_event(storage.source.events.source, storage.source.generation,
                                                    storage.source.pe, execution, context.executor_lane_index, sequence,
                                                    layer.effective_topk, attachment_state, raw_time);
    const auto reserved = sequence ? timeline::try_reserve_device(storage.source.events) : timeline::ReserveResult{};
    if (reserved.status == timeline::ReserveStatus::Reserved)
      timeline::commit_device(storage.source.events, reserved.token, event);
    reserved_sequence = entry && !oversized ? sequence : 0;
    if (entry) {
      *entry = RecordHeader{
          .generation = storage.source.generation,
          .pe = storage.source.pe,
          .oversized = oversized,
          .attachment_sequence = sequence,
          .raw_time_ns = raw_time,
          .key = execution.key,
          .layer_ordinal = execution.layer_ordinal,
          .executor_lane_index = context.executor_lane_index,
          .executor_lease_sequence = execution.executor_lease_sequence,
          .rows = execution.payload_rows,
          .effective_topk = layer.effective_topk,
      };
    }
  }
  group.sync();

  if (reserved_sequence == 0) {
    return;
  }
  const auto record_index = static_cast<std::size_t>(reserved_sequence - 1);
  const auto topk_ids = metadata.topk_ids();
  const auto topk_weights = metadata.topk_weights();
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < live_element_count;
       index += static_cast<std::size_t>(group.size())) {
    view.ids(record_index)[index] = topk_ids[index];
    view.weights(record_index)[index] = topk_weights[index];
  }
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().ffn_routing_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  if (copy_uncertain)
    timeline::fail_routing_transfer("Routing copy retirement is uncertain", "retaining Device and Host owners");
  if (buffer == nullptr) {
    installed = false;
    return;
  }
  const auto empty_storage = Storage{};
  C10_CUDA_CHECK(cudaMemcpyToSymbol(storage, &empty_storage, sizeof(empty_storage)));
  if (buffer != nullptr) {
    C10_CUDA_CHECK(cudaFree(buffer));
  }
  buffer = nullptr;
  device_charge = {};
  pending_copy.reset();
  layout = {};
  installed = false;
}

void collect_drained() {
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  if (!installed || collected)
    return;
  if (buffer == nullptr) {
    collected = true;
    return;
  }
  const auto view = BufferView{buffer, layout};
  const auto bytes = copy_bytes(view.layout().record_capacity, view.layout().element_capacity);
  if (copy_uncertain)
    timeline::fail_routing_transfer("Routing copy cannot retry uncertain ownership", "retaining original copy");
  if (!pending_copy) {
    auto charge = timeline::admit_routing_copy(bytes);
    pending_copy = std::make_shared<RoutingCopy>();
    pending_copy->charge = std::move(charge);
    pending_copy->headers.resize(view.layout().record_capacity);
    pending_copy->records.reserve(view.layout().record_capacity);
  }
  auto &copy = *pending_copy;
  copy_to_host(&copy.state, view.base() + view.layout().state_offset, sizeof(copy.state));
  if (!counts_recorded) {
    constexpr auto upper = std::uint64_t{1} << 63;
    const auto attempts = std::min(copy.state.attempts.lower, upper - 1) + std::min(copy.state.attempts.upper, upper);
    timeline::record_routing_counts(attempts, copy.state.full, copy.state.oversized);
    counts_recorded = true;
  }
  const auto record_count =
      static_cast<std::size_t>(std::min<std::uint64_t>(copy.state.attempts.lower, view.layout().record_capacity));
  if (record_count)
    copy_to_host(copy.headers.data(), view.base() + view.layout().headers_offset, record_count * sizeof(RecordHeader));
  auto &headers = copy.headers;
  auto &records = copy.records;
  records.clear();
  for (auto index = std::size_t{0}; index < record_count; ++index) {
    const auto &header = headers[index];
    if (header.oversized)
      continue;
    TORCH_CHECK(header.key.valid() && header.rows != 0 && header.effective_topk != 0 &&
                    header.rows <= view.layout().element_capacity / header.effective_topk,
                "routing observer record is invalid");
    const auto rows = static_cast<std::int64_t>(header.rows);
    const auto effective_topk = static_cast<std::int64_t>(header.effective_topk);
    auto ids = at::empty({rows, effective_topk}, at::TensorOptions{}.dtype(at::kInt).device(at::kCPU));
    auto weights = at::empty({rows, effective_topk}, at::TensorOptions{}.dtype(at::kFloat).device(at::kCPU));
    const auto element_count = static_cast<std::size_t>(header.rows) * header.effective_topk;
    records.push_back(timeline::RoutingAttachment{
        .sequence = header.attachment_sequence,
        .generation = header.generation,
        .instance_index = header.key.instance_index,
        .invocation_sequence = header.key.invocation_sequence,
        .lane_index = header.executor_lane_index,
        .lease_sequence = header.executor_lease_sequence,
        .layer_ordinal = header.layer_ordinal,
        .pe = header.pe,
        .clock_domain_id = timeline::routing_event_buffer().source.clock_domain_id,
        .raw_publication_time_ns = header.raw_time_ns,
        .topk_ids = std::move(ids),
        .topk_weights = std::move(weights),
    });
    copy_to_host(records.back().topk_ids.data_ptr(), view.ids(index), element_count * sizeof(std::int32_t));
    copy_to_host(records.back().topk_weights.data_ptr(), view.weights(index), element_count * sizeof(float));
  }
  timeline::submit_routing_attachments(std::move(records), std::move(copy.charge));
  pending_copy.reset();
  collected = true;
}

std::size_t allocation_bytes(std::size_t element_capacity) {
  const auto &observer = xpool::debug::options().ffn_routing_observer;
  if (!observer.enable || element_capacity == 0)
    return 0;
  return xpool::utils::checked::sum(sizeof(Storage),
                                    element_capacity == 0
                                        ? std::size_t{0}
                                        : BufferLayout::create(observer.record_capacity, element_capacity).total_bytes);
}

} // namespace xpool::devkit::ffn_routing_observer
