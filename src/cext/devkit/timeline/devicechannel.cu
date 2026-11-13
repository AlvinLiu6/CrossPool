#include <xpool/devkit/timeline/devicechannel.hpp>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

namespace {

XPOOL_KERNEL_FN void initialize_source(BufferView view, FabricRecordingView fabric, TransportRecordingView transport) {
  const auto first = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto stride = static_cast<std::size_t>(blockDim.x * gridDim.x);
  for (auto index = first; index < view.capacity; index += stride)
    view.slots[index] = SlotControl{};
  if (first == 0) {
    *view.pool_state = SlotPoolState{};
    *view.source_state = DeviceSourceState{};
    if (fabric.state)
      *fabric.state = FabricRecordingState{};
  }
  for (auto index = first; index < fabric.instance_count; index += stride) {
    if (fabric.atnagents)
      fabric.atnagents[index] = FabricAtnCurrent{};
    if (fabric.coordinators)
      fabric.coordinators[index] = FabricCoordinatorCurrent{};
  }
  for (auto index = first; index < transport.capacity; index += stride)
    transport.endpoints[index] = TransportRecordingEndpoint{};
  if (fabric.ffnagents)
    for (auto index = first; index < fabric.lane_count; index += stride)
      fabric.ffnagents[index] = FabricFfnCurrent{};
}

} // namespace

std::shared_ptr<DeviceChannel> DeviceChannel::create(std::shared_ptr<MemoryBudget> host,
                                                     std::shared_ptr<MemoryBudget> device, BatchSource source,
                                                     BatchInfo metadata, int device_index, std::size_t slots,
                                                     std::size_t batches, std::size_t events_per_batch,
                                                     std::optional<FabricRecordingConfig> fabric,
                                                     std::size_t transport_capacity) {
  auto result = prepare(std::move(host), std::move(device), source, metadata, device_index, slots, batches,
                        events_per_batch, fabric, transport_capacity);
  result->activate();
  return result;
}

std::shared_ptr<DeviceChannel> DeviceChannel::prepare(std::shared_ptr<MemoryBudget> host,
                                                      std::shared_ptr<MemoryBudget> device, BatchSource source,
                                                      BatchInfo metadata, int device_index, std::size_t slots,
                                                      std::size_t batches, std::size_t events_per_batch,
                                                      std::optional<FabricRecordingConfig> fabric,
                                                      std::size_t transport_capacity) {
  if (!host || !device || source.producer == 0 || source.clock_domain_id == 0 || metadata.metadata_revision == 0 ||
      device_index < 0) {
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool Device channel requires registered source and resource owners");
  }
  try {
    const auto source_layout = DeviceSourceState::layout(slots);
    const auto batch_layout = HostBatchPool::layout(batches, events_per_batch);
    const auto transfer_layout = TransferLayout::for_capacity(events_per_batch);
    auto result = make_budgeted_shared<DeviceChannel>(
        host, MemoryKind::Control, source.producer, ConstructionKey{}, host, device, source, metadata, device_index,
        source_layout, batch_layout, transfer_layout, slots, batches, events_per_batch, fabric, transport_capacity);
    return result;
  } catch (const CollectionError &) {
    throw;
  } catch (const std::invalid_argument &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool Device channel geometry is invalid", std::nullopt, cause.what());
  } catch (const std::overflow_error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool Device channel geometry overflows", std::nullopt, cause.what());
  } catch (const std::exception &cause) {
    throw CollectionError(CollectionErrorCode::Allocation, source.producer, "xpool Device channel setup failed",
                          std::nullopt, cause.what());
  }
}

void DeviceChannel::activate() {
  if (activation_attempted_)
    throw CollectionError(CollectionErrorCode::Allocation, source_.producer,
                          "xpool Device channel activation already attempted");
  activation_attempted_ = true;
  retained_ = shared_from_this();
  try {
    initialize();
    initialized_ = true;
  } catch (...) {
    if (cuda_idle_ && release_storage()) {
      retired_.store(true, std::memory_order_release);
      retained_.reset();
    }
    throw;
  }
}

DeviceChannel::DeviceChannel(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<MemoryBudget> device,
                             BatchSource source, BatchInfo metadata, int device_index, SourceLayout source_layout,
                             BatchPoolLayout batch_layout, TransferLayout transfer_layout, std::size_t slots,
                             std::size_t batches, std::size_t events_per_batch,
                             std::optional<FabricRecordingConfig> fabric, std::size_t transport_capacity)
    : host_(std::move(host)), device_budget_(std::move(device)), source_(source), metadata_(metadata),
      device_index_(device_index), source_layout_(source_layout), batch_layout_(batch_layout),
      transfer_layout_(transfer_layout), fabric_config_(fabric),
      fabric_layout_(fabric ? std::optional{FabricRecordingLayout::for_config(*fabric, source.producer)}
                            : std::nullopt),
      transport_capacity_(transport_capacity),
      transport_bytes_(transport_recording_bytes(transport_capacity, source.producer)) {
  source_charge_ = device_budget_->reserve(MemoryKind::Events, source_layout.total_bytes, source.producer);
  frame_charge_ = device_budget_->reserve(MemoryKind::Transfer, transfer_layout.device_bytes, source.producer);
  pinned_charge_ = host_->reserve(MemoryKind::Transfer, transfer_layout.host_bytes, source.producer);
  if (fabric_layout_)
    fabric_charge_ = device_budget_->reserve(MemoryKind::Control, fabric_layout_->total_bytes, source.producer);
  if (transport_bytes_)
    transport_charge_ = device_budget_->reserve(MemoryKind::Control, transport_bytes_, source.producer);
  batch_storage_ = static_cast<std::byte *>(
      host_->allocate_storage(batch_layout.total_bytes, 256, MemoryKind::Delivery, source.producer));
  try {
    auto *events = reinterpret_cast<Event *>(batch_storage_);
    auto *controls = reinterpret_cast<BatchControl *>(batch_storage_ + batch_layout.control_offset);
    for (auto index = std::size_t{0}; index < batches * events_per_batch; ++index)
      std::construct_at(events + index);
    for (auto index = std::size_t{0}; index < batches; ++index)
      std::construct_at(controls + index);
    auto *state = std::construct_at(reinterpret_cast<BatchPoolState *>(batch_storage_ + batch_layout.state_offset));
    batches_.emplace(source, events_per_batch, std::span{events, batches * events_per_batch},
                     std::span{controls, batches}, *state);
    collector_.emplace(*batches_);
    view_.source = source;
    view_.capacity = slots;
  } catch (...) {
    collector_.reset();
    batches_.reset();
    host_->deallocate_storage(batch_storage_, batch_layout.total_bytes, 256, MemoryKind::Delivery);
    throw;
  }
}

DeviceChannel::~DeviceChannel() {
  transfer_.reset();
  collector_.reset();
  batches_.reset();
  host_->deallocate_storage(batch_storage_, batch_layout_.total_bytes, 256, MemoryKind::Delivery);
}

void DeviceChannel::check(cudaError_t status, CollectionErrorCode code, const char *operation) {
  if (status != cudaSuccess)
    throw CollectionError(code, source_.producer, operation, std::nullopt, cudaGetErrorString(status));
}

void DeviceChannel::initialize() {
  check(cudaSetDevice(device_index_), CollectionErrorCode::Allocation, "xpool initialization device selection failed");
  check(cudaMalloc(reinterpret_cast<void **>(&source_storage_), source_layout_.total_bytes),
        CollectionErrorCode::Allocation, "xpool source allocation failed");
  check(cudaMalloc(reinterpret_cast<void **>(&device_frame_), transfer_layout_.device_bytes),
        CollectionErrorCode::Allocation, "xpool transfer frame allocation failed");
  if (fabric_layout_)
    check(cudaMalloc(reinterpret_cast<void **>(&fabric_storage_), fabric_layout_->total_bytes),
          CollectionErrorCode::Allocation, "xpool Fabric recording allocation failed");
  if (transport_bytes_)
    check(cudaMalloc(reinterpret_cast<void **>(&transport_storage_), transport_bytes_), CollectionErrorCode::Allocation,
          "xpool Transport recording allocation failed");
  check(cudaMallocHost(reinterpret_cast<void **>(&host_frame_), transfer_layout_.host_bytes),
        CollectionErrorCode::Allocation, "xpool pinned frame allocation failed");
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), CollectionErrorCode::Allocation,
        "xpool collection stream creation failed");
  check(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming), CollectionErrorCode::Allocation,
        "xpool copy event creation failed");
  check(cudaEventCreateWithFlags(&reclaimed_, cudaEventDisableTiming), CollectionErrorCode::Allocation,
        "xpool reclaim event creation failed");
  view_.events = reinterpret_cast<Event *>(source_storage_);
  view_.slots = reinterpret_cast<SlotControl *>(source_storage_ + source_layout_.slots.control_offset);
  view_.pool_state = reinterpret_cast<SlotPoolState *>(source_storage_ + source_layout_.slots.state_offset);
  view_.source_state = reinterpret_cast<DeviceSourceState *>(source_storage_ + source_layout_.source_state_offset);
  if (fabric_config_) {
    const auto &config = *fabric_config_;
    const auto &layout = *fabric_layout_;
    fabric_view_ = {
        .buffer = view_,
        .generation = config.generation,
        .pe = config.pe,
        .state = reinterpret_cast<FabricRecordingState *>(fabric_storage_),
        .atnagents =
            config.atnagent ? reinterpret_cast<FabricAtnCurrent *>(fabric_storage_ + layout.current_offset) : nullptr,
        .instance_count = config.instances,
        .record_capacity = config.record_capacity,
        .coordinators = config.coordinator
                            ? reinterpret_cast<FabricCoordinatorCurrent *>(fabric_storage_ + layout.coordinator_offset)
                            : nullptr,
        .ffnagents =
            config.atnagent ? nullptr : reinterpret_cast<FabricFfnCurrent *>(fabric_storage_ + layout.ffn_offset),
        .lane_count = config.lanes,
        .fifo = config.fifo};
  }
  transport_view_ = {view_, reinterpret_cast<TransportRecordingEndpoint *>(transport_storage_), transport_capacity_};
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(65535, (view_.capacity + 255) / 256));
  cuda_idle_ = false;
  initialize_source<<<blocks, 256, 0, stream_>>>(view_, fabric_view_, transport_view_);
  check(cudaGetLastError(), CollectionErrorCode::Allocation, "xpool source initialization launch failed");
  check(cudaStreamSynchronize(stream_), CollectionErrorCode::Allocation, "xpool source initialization wait failed");
  cuda_idle_ = true;
  transfer_.emplace(view_, *collector_, transfer_layout_, device_frame_, host_frame_, stream_, copied_, reclaimed_,
                    fabric_view_);
}

bool DeviceChannel::release_storage() noexcept {
  // No CUDA handle was created: rollback does not need a usable Device context.
  if (!source_storage_ && !device_frame_ && !fabric_storage_ && !transport_storage_ && !host_frame_ && !copied_ &&
      !reclaimed_ && !stream_) {
    source_charge_.reset();
    frame_charge_.reset();
    pinned_charge_.reset();
    fabric_charge_.reset();
    transport_charge_.reset();
    return true;
  }
  if (cleanup_error_ != cudaSuccess)
    return false;
  const auto completed = [this](cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
      cleanup_error_ = status;
      cleanup_operation_ = operation;
      return false;
    }
    return true;
  };
  if (!completed(cudaSetDevice(device_index_), "xpool retirement device selection failed"))
    return false;
  const auto free = [&completed](auto &pointer, auto function, const char *operation) {
    if (pointer != nullptr) {
      if (!completed(function(pointer), operation))
        return false;
      pointer = nullptr;
    }
    return true;
  };
  if (!free(transport_storage_, cudaFree, "xpool Transport recording retirement failed") ||
      !free(fabric_storage_, cudaFree, "xpool Fabric recording retirement failed") ||
      !free(source_storage_, cudaFree, "xpool source retirement failed") ||
      !free(device_frame_, cudaFree, "xpool transfer frame retirement failed") ||
      !free(host_frame_, cudaFreeHost, "xpool pinned frame retirement failed") ||
      !free(copied_, cudaEventDestroy, "xpool copy event retirement failed") ||
      !free(reclaimed_, cudaEventDestroy, "xpool reclaim event retirement failed") ||
      !free(stream_, cudaStreamDestroy, "xpool collection stream retirement failed"))
    return false;
  source_charge_.reset();
  frame_charge_.reset();
  pinned_charge_.reset();
  fabric_charge_.reset();
  view_ = {};
  fabric_view_ = {};
  transport_view_ = {};
  transport_charge_.reset();
  return true;
}

bool DeviceChannel::tick(bool producer_retired, MetadataResolver resolver, ReconstructionSink reconstruction) {
  if (!initialized_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool Device channel is not activated");
  if (retired())
    return false;
  if (transfer_uncertain_)
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool device channel retirement is uncertain");
  try {
    check(cudaSetDevice(device_index_), CollectionErrorCode::Transfer, "xpool collection device selection failed");
    cuda_idle_ = false;
    const auto progress = transfer_->tick(metadata_, producer_retired, stopping_.load(std::memory_order_acquire),
                                          resolver, reconstruction);
    if (transfer_->closed()) {
      cuda_idle_ = true;
      if (!release_storage()) {
        throw CollectionError(CollectionErrorCode::Transfer, source_.producer, cleanup_operation_, std::nullopt,
                              cudaGetErrorString(cleanup_error_));
      }
      retired_.store(true, std::memory_order_release);
      retained_.reset();
    }
    return progress;
  } catch (const CollectionError &error) {
    if (error.code() == CollectionErrorCode::Transfer || transfer_->in_flight()) {
      transfer_uncertain_ = true;
      collector_->fail_transfer();
    }
    throw;
  } catch (...) {
    transfer_uncertain_ = true;
    collector_->fail_transfer();
    throw;
  }
}

} // namespace xpool::devkit::timeline
