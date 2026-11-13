#include <xpool/transport/instance.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <cuda_runtime_api.h>

#include <xpool/hooks.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kEndpointStartupTimeout = std::chrono::seconds{60};
constexpr auto kEndpointStartupPollInterval = std::chrono::milliseconds{1};
constexpr auto kGraphRetirementTimeout = std::chrono::seconds{1};

} // namespace

// One Host-budgeted callback target per observed Endpoint. CUDA owns the
// Graph/clone/executable references; no application map grows with captures.
// The self-reference keeps the callback target and its ledger alive even after
// failed initialization or process-static Attachment destruction.
class InstanceRankRuntime::Attachment::GraphUse {
  static_assert(std::atomic<bool>::is_always_lock_free);

public:
  void initialize(const std::shared_ptr<GraphUse> &self) {
    retained_ = self;
    C10_CUDA_CHECK(cudaUserObjectCreate(&object_, this, &GraphUse::released, 1, cudaUserObjectNoDestructorSync));
  }

  void capture(cudaStream_t stream) const {
    auto status = cudaStreamCaptureStatusNone;
    auto graph = cudaGraph_t{};
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph));
    TORCH_CHECK(status != cudaStreamCaptureStatusInvalidated, "xpool request stream capture is invalidated");
    if (status == cudaStreamCaptureStatusActive)
      C10_CUDA_CHECK(cudaGraphRetainUserObject(graph, object_, 1, 0));
  }

  void retire() {
    // Attachment's mutex closes submit/attach admission throughout this call.
    // Do not release the creator reference twice, even if a later wait fails.
    C10_CUDA_CHECK(cudaUserObjectRelease(object_, 1));
    object_ = nullptr;
    const auto result = xpool::utils::wait::until(
        std::chrono::steady_clock::now() + kGraphRetirementTimeout,
        [&] { return released_.load(std::memory_order_acquire); }, kEndpointStartupPollInterval);
    TORCH_CHECK(result == xpool::utils::wait::Status::Ready,
                "xpool captured Graph references did not retire before the bounded deadline; "
                "destroy all Graph and GraphExec owners before detach; process retirement required");
    retained_.reset();
  }

private:
  static void CUDART_CB released(void *target) {
    // CUDA's shared callback thread must neither block nor call CUDA. This
    // store is its final access; cleanup belongs to the waiting Host owner.
    static_cast<GraphUse *>(target)->released_.store(true, std::memory_order_release);
  }

  cudaUserObject_t object_ = nullptr;
  std::atomic<bool> released_{false};
  std::shared_ptr<GraphUse> retained_;
};

void InstanceRankRuntime::Attachment::attach(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle,
                                             std::shared_ptr<ObservationPool> observation_pool,
                                             const std::optional<ObservationRegion> &descriptor,
                                             std::uint64_t generation, std::uint64_t owner_producer) {
  TORCH_CHECK(!arena_.uncertain(), "xpool uncertain attachment requires process retirement");
  validate_observation_region(descriptor, observation_pool != nullptr, generation, owner_producer);
  TORCH_CHECK(!prepared_pool_ || prepared_pool_ == observation_pool,
              "xpool instance cannot replace its prepared observation pool or policy");
  if (arena_) {
    TORCH_CHECK(ready_ && !arena_.uncertain() && matches(instance_index, rank, handle) &&
                    arena_.observation_region() == descriptor && arena_.observation_pool() == observation_pool,
                "xpool instance retains a different or incomplete transport attachment");
    return;
  }
  if (observation_pool) {
    TORCH_CHECK(endpoint_owner_ == 0 || endpoint_owner_ == descriptor->owner_producer,
                "xpool instance cannot replace its prepared Endpoint owner");
    TORCH_CHECK(descriptor->endpoint_sequence > last_endpoint_sequence_,
                "xpool reattachment requires a fresh Endpoint after observed detach");
    if (!prepared_pool_) {
      control_ =
          observation_pool->host_budget()->reserve(xpool::devkit::timeline::MemoryKind::Control,
                                                   sizeof(*this) - sizeof(arena_), observation_pool->source().producer);
      prepared_pool_ = observation_pool;
    }
    endpoint_owner_ = descriptor->owner_producer;
    last_endpoint_sequence_ = descriptor->endpoint_sequence;
  }
  handle_ = handle;
  try {
    arena_.open(handle, std::move(observation_pool), descriptor, generation, owner_producer);
    if (prepared_pool_) {
      graph_use_ = xpool::devkit::timeline::make_budgeted_shared<GraphUse>(prepared_pool_->host_budget(),
                                                                           xpool::devkit::timeline::MemoryKind::Control,
                                                                           prepared_pool_->source().producer);
      const auto device_guard = c10::cuda::CUDAGuard{arena_.device()};
      graph_use_->initialize(graph_use_);
    }
    const auto &layout = arena_.layout();
    TORCH_CHECK(layout.instance_index == instance_index && layout.instance_rank == rank,
                "xpool transport arena identity does not match the attaching instance process");
    const auto result = xpool::utils::wait::until(
        std::chrono::steady_clock::now() + kEndpointStartupTimeout,
        [&] {
          const auto status = arena_.mailbox_status();
          if (status == MailboxStatus::Idle) {
            return true;
          }
          TORCH_CHECK(status == MailboxStatus::Dormant,
                      "xpool Transport endpoint entered an invalid state before first use");
          return false;
        },
        kEndpointStartupPollInterval);
    TORCH_CHECK(result == xpool::utils::wait::Status::Ready,
                "xpool Transport endpoint startup exceeded the bounded deadline");
    xpool::hooks::TransportEndpointOpenPostEvent::hooks({.device = arena_.device(),
                                                         .arena = arena_.view(),
                                                         .layout = arena_.layout(),
                                                         .site = xpool::hooks::TransportEndpointSite::Instance});
    ready_ = true;
  } catch (...) {
    arena_.quarantine();
    throw;
  }
}

void InstanceRankRuntime::Attachment::detach() {
  TORCH_CHECK(!arena_.uncertain(), "xpool uncertain Transport attachment requires process retirement");
  if (!arena_)
    return;
  try {
    const auto device_guard = c10::cuda::CUDAGuard{arena_.device()};
    if (graph_use_)
      graph_use_->retire();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    if (ready_)
      xpool::hooks::TransportEndpointClosePreEvent::hooks({.device = arena_.device(),
                                                           .arena = arena_.view(),
                                                           .layout = arena_.layout(),
                                                           .site = xpool::hooks::TransportEndpointSite::Instance});
    arena_.destroy(true, false);
    graph_use_.reset();
    handle_ = {};
    ready_ = false;
  } catch (...) {
    arena_.quarantine();
    throw;
  }
}

xpool::ffn::ResultCode InstanceRankRuntime::Attachment::read_generation_failure() const {
  try {
    c10::cuda::CUDAGuard device_guard(arena_.device());
    return arena_.read_generation_failure();
  } catch (...) {
    arena_.quarantine();
    throw;
  }
}

void InstanceRankRuntime::Attachment::submit(const at::Tensor &hidden_states,
                                             const std::optional<at::Tensor> &dp_rank_payload_rows,
                                             const at::Tensor &output,
                                             const xpool::transport::RequestMetadata &request_metadata) const {
  const auto &layout = arena_.layout();
  const auto payload_rows = c10::checked_convert<std::size_t>(hidden_states.size(0), "hidden-state rows");
  const auto dp_rank_payload_rows_present =
      dp_rank_payload_rows.has_value() && dp_rank_payload_rows->defined() && dp_rank_payload_rows->numel() != 0;
  TORCH_CHECK(request_metadata.valid(), "xpool FFN shim received invalid request metadata");
  TORCH_CHECK(payload_rows != 0 && payload_rows <= layout.payload_row_capacity,
              "xpool FFN request row count exceeds Transport capacity");
  if (layout.atn_dp_size == 1) {
    TORCH_CHECK(request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::None && !dp_rank_payload_rows_present,
                "xpool DP-one request must use NONE without a per-rank row vector");
  } else {
    TORCH_CHECK(request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::UniformByRank ||
                    request_metadata.dp_row_layout == xpool::ffn::DpRowLayout::PackedByRank,
                "xpool DP request requires UNIFORM_BY_RANK or PACKED_BY_RANK");
    TORCH_CHECK(dp_rank_payload_rows_present, "xpool DP request requires one physical row value per attention DP rank");
  }
  TORCH_CHECK(c10::checked_convert<std::size_t>(hidden_states.size(1), "hidden-state width") == layout.hidden_size,
              "xpool FFN shim hidden size does not match the attached Transport arena");
  TORCH_CHECK(hidden_states.scalar_type() == layout.payload_dtype,
              "xpool FFN shim dtype does not match the attached Transport arena");
  if (dp_rank_payload_rows_present) {
    const auto &rank_payload_rows = *dp_rank_payload_rows;
    TORCH_CHECK(c10::checked_convert<std::size_t>(rank_payload_rows.numel(), "DP-rank row vector length") ==
                    layout.atn_dp_size,
                "xpool FFN shim DP-rank payload rows must have one entry per attention DP rank");
  }

  // Geometry errors above are safe to correct. CUDA queries and launch can
  // surface earlier asynchronous failures and must retain a sticky stop fact.
  auto device = c10::DeviceIndex{0};
  try {
    device = arena_.device();
  } catch (...) {
    arena_.quarantine();
    throw;
  }
  TORCH_CHECK(hidden_states.get_device() == device,
              "xpool FFN shim hidden states must be on the attached Transport arena device");
  try {
    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    const auto stream = at::cuda::getCurrentCUDAStream(hidden_states.device().index());
    if (graph_use_)
      graph_use_->capture(stream);
    const Request request{
        stream, arena_.view(), hidden_states, dp_rank_payload_rows, output, request_metadata, arena_.observation_view(),
    };
    launch_request_kernel(request);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } catch (...) {
    arena_.quarantine();
    throw;
  }
}

void InstanceRankRuntime::attach_arena(std::size_t instance_index, std::size_t rank, const ArenaHandle &handle,
                                       std::shared_ptr<ObservationPool> observation_pool,
                                       const std::optional<ObservationRegion> &descriptor, std::uint64_t generation,
                                       std::uint64_t owner_producer) {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.attach(instance_index, rank, handle, std::move(observation_pool), descriptor, generation, owner_producer);
}

void InstanceRankRuntime::detach_arena() {
  std::lock_guard<std::mutex> lock(mutex_);
  attachment_.detach();
}

xpool::ffn::ResultCode InstanceRankRuntime::read_generation_failure() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(attachment_, "xpool transport failure read requires an attached arena");
  return attachment_.read_generation_failure();
}

void InstanceRankRuntime::submit(const at::Tensor &hidden_states, const std::optional<at::Tensor> &dp_rank_payload_rows,
                                 const at::Tensor &output, const xpool::transport::RequestMetadata &request_metadata) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(attachment_, "xpool ffn_shim has no attached instance transport arena");
  attachment_.submit(hidden_states, dp_rank_payload_rows, output, request_metadata);
}

} // namespace xpool::transport
