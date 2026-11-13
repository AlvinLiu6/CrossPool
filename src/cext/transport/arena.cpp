#include <xpool/transport/arena.hpp>

#include <cstddef>
#include <cstdint>

#include <ATen/Functions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>
#include <cuda_runtime_api.h>

namespace xpool::transport {

Arena Arena::create(c10::DeviceIndex device, const ArenaLayout &layout) {
  auto result = Arena{};
  result.allocate(device, layout);
  return result;
}

Arena Arena::from_handle(const ArenaHandle &handle) {
  auto result = Arena{};
  result.open(handle);
  return result;
}

void Arena::allocate(c10::DeviceIndex device, const ArenaLayout &layout, std::shared_ptr<ObservationPool> pool,
                     std::uint64_t generation) {
  TORCH_CHECK(!*this && !uncertain_, "xpool cannot replace a retained or uncertain Transport arena");
  layout.validate();
  if (pool) {
    TORCH_CHECK(pool->device_index() == device && generation != 0,
                "xpool observed Arena requires its prepared Device and Generation");
    TORCH_CHECK(!observation_host_ || observation_host_ == pool->host_budget(),
                "xpool observed Arena control cannot change Host ledgers");
    if (!observation_control_.bytes()) {
      observation_control_ = pool->host_budget()->reserve(xpool::devkit::timeline::MemoryKind::Control, sizeof(*this),
                                                          pool->source().producer);
      observation_host_ = pool->host_budget();
    }
    observation_pool_ = std::move(pool);
  }
  kind_ = Kind::Owned;
  layout_ = layout;
  try {
    if (observation_pool_)
      observation_ = observation_pool_->allocate(generation);
    const auto device_guard = c10::cuda::CUDAGuard{device};
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&base_), layout.header.total_bytes));
    C10_CUDA_CHECK(cudaMemset(base_, 0, layout.header.total_bytes));
    C10_CUDA_CHECK(cudaMemcpy(base_, &layout, sizeof(layout), cudaMemcpyHostToDevice));
    const auto mailbox = Mailbox{.status = MailboxStatus::Dormant,
                                 .result_code = xpool::ffn::ResultCode::ProtocolMismatch,
                                 .payload_rows = 0,
                                 .request = {}};
    C10_CUDA_CHECK(cudaMemcpy(base_ + layout.mailbox_offset, &mailbox, sizeof(mailbox), cudaMemcpyHostToDevice));
  } catch (...) {
    if (observation_pool_)
      quarantine();
    throw;
  }
}

void Arena::open(const ArenaHandle &handle, std::shared_ptr<ObservationPool> pool,
                 const std::optional<ObservationRegion> &descriptor, std::uint64_t generation,
                 std::uint64_t owner_producer) {
  TORCH_CHECK(!*this && !uncertain_, "xpool cannot replace a retained or uncertain Transport mapping");
  validate_observation_region(descriptor, pool != nullptr, generation, owner_producer);
  if (pool) {
    TORCH_CHECK(descriptor->allocation_bytes == ObservationPool::allocation_bytes,
                "xpool observation IPC extent differs from Native preparation");
    TORCH_CHECK(!observation_host_ || observation_host_ == pool->host_budget(),
                "xpool observed Arena control cannot change Host ledgers");
    if (!observation_control_.bytes()) {
      observation_control_ = pool->host_budget()->reserve(xpool::devkit::timeline::MemoryKind::Control, sizeof(*this),
                                                          pool->source().producer);
      observation_host_ = pool->host_budget();
    }
    observation_pool_ = std::move(pool);
  }
  kind_ = Kind::Attached;
  try {
    auto guard = std::optional<c10::cuda::CUDAGuard>{};
    if (observation_pool_)
      guard.emplace(static_cast<c10::DeviceIndex>(observation_pool_->device_index()));
    if (observation_pool_)
      observation_ = observation_pool_->attach(descriptor, true, generation, owner_producer);
    C10_CUDA_CHECK(
        cudaIpcOpenMemHandle(reinterpret_cast<void **>(&base_), handle.value(), cudaIpcMemLazyEnablePeerAccess));
    C10_CUDA_CHECK(cudaMemcpy(&layout_, base_, sizeof(layout_), cudaMemcpyDeviceToHost));
    layout_.validate();
  } catch (...) {
    if (observation_pool_)
      quarantine();
    throw;
  }
}

std::optional<ObservationRegion> Arena::observation_region() const {
  TORCH_CHECK(!uncertain_, "xpool observed Arena completion is uncertain");
  return observation_ ? std::optional{observation_->descriptor()} : std::nullopt;
}

ObservationView Arena::observation_view() const {
  TORCH_CHECK(!uncertain_, "xpool uncertain Arena cannot lend observation state");
  if (!observation_pool_)
    return {};
  TORCH_CHECK(observation_ && !observation_->retired(), "xpool Arena observation state is incomplete or retired");
  const auto &descriptor = observation_->descriptor();
  return {observation_->word(), observation_->operation_state(), descriptor.owner_producer,
          descriptor.endpoint_sequence, descriptor.generation};
}

void Arena::quarantine() const {
  uncertain_ = true;
  if (observation_pool_)
    observation_pool_->stop();
}

c10::DeviceIndex Arena::device() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot query an empty transport arena device");
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, base_));
  return c10::checked_convert<c10::DeviceIndex>(attributes.device, "CUDA pointer device");
}

void Arena::destroy(bool local_users_retired, bool peer_mappings_detached) {
  if (!*this)
    return;
  if (observation_pool_) {
    TORCH_CHECK(!uncertain_, "xpool observed Arena requires process retirement after uncertain CUDA completion");
    TORCH_CHECK(local_users_retired && (kind_ == Kind::Attached || peer_mappings_detached),
                "xpool observed Arena release lacks local or peer retirement proof");
  }
  try {
    auto guard = std::optional<c10::cuda::CUDAGuard>{};
    if (observation_pool_)
      guard.emplace(static_cast<c10::DeviceIndex>(observation_pool_->device_index()));
    if (base_) {
      if (kind_ == Kind::Owned)
        C10_CUDA_CHECK(cudaFree(base_));
      else
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(base_));
      base_ = nullptr;
    }
    if (observation_pool_)
      TORCH_CHECK(observation_pool_->retire(observation_, local_users_retired, peer_mappings_detached),
                  "xpool observation mapping could not retire");
  } catch (...) {
    if (observation_pool_)
      quarantine();
    throw;
  }
  layout_ = {};
  observation_.reset();
  observation_pool_.reset();
}

ArenaHandle Arena::handle() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot export a CUDA IPC handle for an empty arena");
  TORCH_CHECK(kind_ == Kind::Owned, "xpool can only export handles for owned transport arenas");
  ArenaHandle result{};
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&result.value(), base_));
  return result;
}

ArenaState Arena::state() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read state from an empty transport arena");
  ArenaState result{};
  C10_CUDA_CHECK(cudaMemcpy(&result, base_ + layout_.header.state_offset, sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}

MailboxStatus Arena::mailbox_status() const {
  TORCH_CHECK(base_ != nullptr, "xpool cannot read mailbox state from an empty transport arena");
  auto status = std::uint32_t{0};
  C10_CUDA_CHECK(cudaMemcpy(&status, base_ + layout_.mailbox_offset + offsetof(Mailbox, status), sizeof(status),
                            cudaMemcpyDeviceToHost));
  TORCH_CHECK(status <= static_cast<std::uint32_t>(MailboxStatus::Closed),
              "xpool transport mailbox contains an invalid status value");
  return static_cast<MailboxStatus>(status);
}

} // namespace xpool::transport
