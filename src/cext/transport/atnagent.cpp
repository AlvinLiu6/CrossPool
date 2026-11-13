#include <xpool/transport/atnagent.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory_resource>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffn.hpp>
#include <xpool/hooks.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/device.hpp>
#include <xpool/utils/wait.hpp>

namespace xpool::transport {

namespace {

constexpr auto kResidentStartupTimeout = std::chrono::seconds{60};
constexpr auto kResidentStartupPollInterval = std::chrono::milliseconds{1};

struct ResidentLayout {
  std::size_t arenas;
  std::size_t state;
  std::size_t observations;
  std::size_t total;
};

ResidentLayout resident_layout(std::size_t count, bool observed) {
  TORCH_CHECK(count != 0, "xpool Resident requires a nonempty Arena set");
  auto arenas = xpool::utils::checked::prod(count, sizeof(ArenaView));
  auto state = sizeof(ResidentState);
  auto observations = observed ? xpool::utils::checked::prod(count, sizeof(ObservationView)) : std::size_t{0};
  if (observed) {
    arenas = xpool::utils::checked::align_up(arenas, xpool::arena::kAllocationAlignment);
    state = xpool::utils::checked::align_up(state, xpool::arena::kAllocationAlignment);
    observations = xpool::utils::checked::align_up(observations, xpool::arena::kAllocationAlignment);
  }
  return {arenas, state, observations, xpool::utils::checked::sum(arenas, state, observations)};
}

} // namespace

std::size_t resident_observation_device_bytes(std::size_t arena_count) {
  try {
    return resident_layout(arena_count, true).total;
  } catch (const c10::Error &) {
    throw xpool::devkit::timeline::CollectionError(xpool::devkit::timeline::CollectionErrorCode::ResourceBudget, 0,
                                                   "xpool Resident observation demand is zero or overflows");
  }
}

AtnAgentRuntime::Resident::Resident(c10::DeviceIndex device, const std::map<ArenaHandle, Arena> &arenas,
                                    std::shared_ptr<ObservationPool> pool)
    : device_(device), pool_(std::move(pool)),
      host_arenas_(pool_ ? pool_->host_resource() : std::pmr::get_default_resource()),
      host_observations_(pool_ ? pool_->host_resource() : std::pmr::get_default_resource()) {
  const auto layout = resident_layout(arenas.size(), pool_ != nullptr);
  arena_bytes_ = layout.arenas;
  state_bytes_ = layout.state;
  observation_bytes_ = layout.observations;
  if (pool_) {
    device_control_ = pool_->device_budget()->reserve(xpool::devkit::timeline::MemoryKind::Control, layout.total,
                                                      pool_->source().producer);
  }
  auto ordered = std::pmr::vector<const Arena *>{host_arenas_.get_allocator().resource()};
  ordered.reserve(arenas.size());
  for (const auto &entry : arenas) {
    TORCH_CHECK(entry.second.observation_pool() == pool_, "xpool Resident cannot mix observation policies or pools");
    ordered.push_back(&entry.second);
  }
  std::ranges::sort(ordered, {}, [](const Arena *arena) { return arena->layout().instance_index; });
  host_arenas_.reserve(ordered.size());
  if (pool_)
    host_observations_.reserve(ordered.size());
  for (const auto *arena : ordered) {
    host_arenas_.push_back(arena->view());
    if (pool_)
      host_observations_.push_back(arena->observation_view());
  }
}

void AtnAgentRuntime::Resident::initialize(xpool::fabric::ArenaView fabric_arena) {
  TORCH_CHECK(!initialized_ && !uncertain_ && !released_, "xpool Resident initialization cannot be retried");
  retained_ = shared_from_this();
  uncertain_ = true;
  try {
    const auto device_guard = c10::cuda::CUDAGuard{device_};
    // Assign CUDA outputs directly into retained owners before checking status.
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&arenas_), arena_bytes_));
    C10_CUDA_CHECK(
        cudaMemcpy(arenas_, host_arenas_.data(), host_arenas_.size() * sizeof(ArenaView), cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state_), state_bytes_));
    C10_CUDA_CHECK(cudaMemcpy(state_, &initial_state_, sizeof(initial_state_), cudaMemcpyHostToDevice));
    if (pool_) {
      C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&observations_), observation_bytes_));
      C10_CUDA_CHECK(cudaMemcpy(observations_, host_observations_.data(),
                                host_observations_.size() * sizeof(ObservationView), cudaMemcpyHostToDevice));
    }
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&control_stream_, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&resident_stream_, cudaStreamNonBlocking));
    launch_resident_kernel(cuda::std::span<const ArenaView>{arenas_, host_arenas_.size()}, fabric_arena, state_,
                           resident_stream_,
                           pool_ ? cuda::std::span<const ObservationView>{observations_, host_observations_.size()}
                                 : cuda::std::span<const ObservationView>{});
    initialized_ = true;
    uncertain_ = false;
  } catch (...) {
    quarantine();
    throw;
  }
}

void AtnAgentRuntime::Resident::quarantine() const {
  uncertain_ = true;
  if (pool_)
    pool_->stop();
}

bool AtnAgentRuntime::Resident::pending() const {
  if (released_)
    return false;
  TORCH_CHECK(initialized_ && !uncertain_, "xpool uncertain Resident requires process retirement");
  try {
    const auto device_guard = c10::cuda::CUDAGuard{device_};
    const auto pending = [](cudaStream_t stream) {
      const auto status = cudaStreamQuery(stream);
      if (status == cudaErrorNotReady)
        return true;
      C10_CUDA_CHECK(status);
      return false;
    };
    return pending(control_stream_) || pending(resident_stream_);
  } catch (...) {
    quarantine();
    throw;
  }
}

void AtnAgentRuntime::Resident::request_drain() {
  TORCH_CHECK(initialized_ && !uncertain_, "xpool uncertain Resident requires process retirement");
  if (host_drain_requested_)
    return;
  try {
    const auto device_guard = c10::cuda::CUDAGuard{device_};
    auto *address = reinterpret_cast<std::uint32_t *>(reinterpret_cast<std::uint8_t *>(state_) +
                                                      offsetof(ResidentState, drain_requested));
    xpool::utils::device::check_driver_result(cuStreamWriteValue32(reinterpret_cast<CUstream>(control_stream_),
                                                                   reinterpret_cast<CUdeviceptr>(address), 1,
                                                                   CU_STREAM_WRITE_VALUE_DEFAULT));
    host_drain_requested_ = true;
  } catch (...) {
    quarantine();
    throw;
  }
}

void AtnAgentRuntime::Resident::release() {
  if (released_)
    return;
  TORCH_CHECK(initialized_ && !uncertain_ && host_drain_requested_,
              "xpool Resident release requires successful initialization and a proven drain");
  TORCH_CHECK(!pending(), "xpool Resident cannot release in-flight users");
  try {
    const auto device_guard = c10::cuda::CUDAGuard{device_};
    C10_CUDA_CHECK(cudaStreamDestroy(control_stream_));
    control_stream_ = nullptr;
    C10_CUDA_CHECK(cudaStreamDestroy(resident_stream_));
    resident_stream_ = nullptr;
    C10_CUDA_CHECK(cudaFree(arenas_));
    arenas_ = nullptr;
    C10_CUDA_CHECK(cudaFree(state_));
    state_ = nullptr;
    if (observations_) {
      C10_CUDA_CHECK(cudaFree(observations_));
      observations_ = nullptr;
    }
    device_control_.reset();
    released_ = true;
    retained_.reset();
  } catch (...) {
    quarantine();
    throw;
  }
}

Arena &AtnAgentRuntime::arena(const ArenaHandle &handle) {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(), "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

const Arena &AtnAgentRuntime::arena(const ArenaHandle &handle) const {
  const auto iter = arenas_.find(handle);
  TORCH_CHECK(iter != arenas_.end(), "xpool Transport arena handle is unknown or already destroyed");
  return iter->second;
}

bool AtnAgentRuntime::generation_failed() const {
  try {
    for (const auto &entry : arenas_) {
      const auto code = entry.second.read_generation_failure();
      if (code != xpool::ffn::ResultCode::Ok) {
        return true;
      }
    }
  } catch (...) {
    if (resident_)
      resident_->quarantine();
    throw;
  }
  return false;
}

ArenaHandle AtnAgentRuntime::create_arena(c10::DeviceIndex device, std::size_t instance_index,
                                          std::size_t instance_rank, std::size_t payload_row_capacity,
                                          std::size_t hidden_size, c10::ScalarType payload_dtype,
                                          std::size_t atn_tp_rank, std::size_t atn_tp_size, std::size_t atn_dp_rank,
                                          std::size_t atn_dp_size, std::shared_ptr<ObservationPool> observation_pool,
                                          std::uint64_t generation) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_, "xpool cannot create a Transport arena after Resident activation");
  TORCH_CHECK(!device_.has_value() || *device_ == device, "xpool AtnAgent Transport arenas must share one device");
  for (const auto &entry : arenas_) {
    TORCH_CHECK(entry.second.layout().instance_index != instance_index,
                "xpool AtnAgent already owns a Transport arena for this instance index");
  }

  const auto layout = ArenaLayout::create(instance_index, instance_rank, atn_tp_rank, atn_tp_size, atn_dp_rank,
                                          atn_dp_size, payload_row_capacity, hidden_size, payload_dtype);
  if (observation_pool) {
    TORCH_CHECK(!pending_arena_ && !pending_arena_.uncertain(), "xpool Transport retains an incomplete Arena creation");
    if (!pending_control_.bytes()) {
      pending_control_ = observation_pool->host_budget()->reserve(xpool::devkit::timeline::MemoryKind::Control,
                                                                  sizeof(pending_arena_) + sizeof(pending_host_) +
                                                                      sizeof(pending_control_),
                                                                  observation_pool->source().producer);
      pending_host_ = observation_pool->host_budget();
    }
    TORCH_CHECK(pending_host_ == observation_pool->host_budget(), "xpool Transport cannot replace its Host ledger");
    try {
      pending_arena_.allocate(device, layout, std::move(observation_pool), generation);
      xpool::hooks::TransportEndpointOpenPostEvent::hooks({.device = device,
                                                           .arena = pending_arena_.view(),
                                                           .layout = pending_arena_.layout(),
                                                           .site = xpool::hooks::TransportEndpointSite::AtnAgent});
      const auto handle = pending_arena_.handle();
      TORCH_CHECK(arenas_.try_emplace(handle, std::move(pending_arena_)).second,
                  "xpool AtnAgent created a duplicate Transport arena handle");
      device_ = device;
      return handle;
    } catch (...) {
      pending_arena_.quarantine();
      throw;
    }
  }
  auto arena = Arena::create(device, layout);
  xpool::hooks::TransportEndpointOpenPostEvent::hooks({.device = device,
                                                       .arena = arena.view(),
                                                       .layout = arena.layout(),
                                                       .site = xpool::hooks::TransportEndpointSite::AtnAgent});
  const auto handle = arena.handle();
  TORCH_CHECK(arenas_.try_emplace(handle, std::move(arena)).second,
              "xpool AtnAgent created a duplicate Transport arena handle");
  device_ = device;
  return handle;
}

std::optional<ObservationRegion> AtnAgentRuntime::observation_region(const ArenaHandle &handle) const {
  const auto lock = std::lock_guard{mutex_};
  return arena(handle).observation_region();
}

void AtnAgentRuntime::activate() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_, "xpool Transport Resident is already active or terminal");
  TORCH_CHECK(device_.has_value() && !arenas_.empty(), "xpool Transport Resident requires a non-empty arena set");

  const auto fabric_arena = xpool::fabric::Runtime::singleton().arena();
  const auto pool = arenas_.begin()->second.observation_pool();
  resident_ = pool ? xpool::devkit::timeline::make_budgeted_shared<Resident>(
                         pool->host_budget(), xpool::devkit::timeline::MemoryKind::Control, pool->source().producer,
                         *device_, arenas_, pool)
                   : std::make_shared<Resident>(*device_, arenas_, pool);
  resident_->initialize(fabric_arena);

  const auto ready = [&] {
    auto ready = true;
    for (const auto &entry : arenas_) {
      auto status = MailboxStatus::Dormant;
      try {
        status = entry.second.mailbox_status();
      } catch (...) {
        resident_->quarantine();
        throw;
      }
      if (status == MailboxStatus::Dormant) {
        ready = false;
      } else {
        TORCH_CHECK(status == MailboxStatus::Idle,
                    "xpool Transport Resident published an invalid startup mailbox state");
      }
    }
    return ready;
  };
  const auto result = xpool::utils::wait::until(
      std::chrono::steady_clock::now() + kResidentStartupTimeout, ready, [&] { return !resident_->pending(); },
      kResidentStartupPollInterval);
  switch (result) {
  case xpool::utils::wait::Status::Pending:
    break;
  case xpool::utils::wait::Status::Ready:
    return;
  case xpool::utils::wait::Status::Cancelled:
    TORCH_CHECK(false, "xpool Transport Resident completed before publishing every endpoint ready");
  case xpool::utils::wait::Status::TimedOut:
    TORCH_CHECK(false, "xpool Transport Resident startup exceeded the bounded deadline");
  }
  xpool::abort();
}

void AtnAgentRuntime::check_health() const {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(resident_ && !resident_->released(), "xpool Transport Resident has not been activated or was released");
  if (resident_->pending()) {
    return;
  }
  TORCH_CHECK(resident_->host_drain_requested() || generation_failed(),
              "xpool Transport Resident completed unexpectedly");
}

void AtnAgentRuntime::drain_async() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(), "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_, "xpool Transport Resident has not been activated");
  if (resident_->released()) {
    TORCH_CHECK(resident_->host_drain_requested(),
                "xpool Transport Resident was released without a host drain request");
    return;
  }
  resident_->request_drain();
}

bool AtnAgentRuntime::drain_pending() {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!arenas_.empty(), "xpool Transport Resident is unavailable after arena destruction");
  TORCH_CHECK(resident_ && resident_->host_drain_requested(), "xpool Transport Resident drain has not been started");
  if (resident_->released()) {
    return false;
  }
  if (resident_->pending()) {
    return true;
  }
  resident_->release();
  return false;
}

void AtnAgentRuntime::destroy_arenas(std::span<const ArenaHandle> handles, bool peer_mappings_detached) {
  const auto lock = std::lock_guard<std::mutex>{mutex_};
  TORCH_CHECK(!resident_ || resident_->released(),
              "xpool Transport arenas require preactivation rollback or completed Resident drain");

  auto distinct = std::set<ArenaHandle>{};
  for (const auto &handle : handles) {
    TORCH_CHECK(distinct.insert(handle).second, "xpool Transport arena collection contains a duplicate handle");
    const auto &resolved = arena(handle);
    TORCH_CHECK(!resolved.observation_region() || peer_mappings_detached,
                "xpool observed Transport release requires actual peer detach");
  }

  auto failures = std::vector<std::string>{};
  for (const auto &handle : handles) {
    try {
      auto &resolved = arena(handle);
      xpool::hooks::TransportEndpointClosePreEvent::hooks({.device = *device_,
                                                           .arena = resolved.view(),
                                                           .layout = resolved.layout(),
                                                           .site = xpool::hooks::TransportEndpointSite::AtnAgent});
      resolved.destroy(true, peer_mappings_detached);
      arenas_.erase(handle);
    } catch (const std::exception &error) {
      arena(handle).quarantine();
      failures.push_back(handle.encode() + ": " + error.what());
    } catch (...) {
      arena(handle).quarantine();
      failures.push_back(handle.encode() + ": unknown native cleanup error");
    }
  }
  if (!resident_ && arenas_.empty()) {
    device_.reset();
  }
  if (!failures.empty()) {
    auto message = std::ostringstream{};
    message << "xpool failed to destroy Transport arenas";
    for (const auto &failure : failures) {
      message << "\n- " << failure;
    }
    TORCH_CHECK(false, message.str());
  }
}

} // namespace xpool::transport
