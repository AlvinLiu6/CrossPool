#include <xpool/transport/observationpool.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/TypeCast.h>

namespace xpool::transport {
namespace timeline = xpool::devkit::timeline;

namespace {

c10::cuda::CUDAGuard guard_device(int device_index, std::uint64_t producer) {
  try {
    return c10::cuda::CUDAGuard(c10::checked_convert<c10::DeviceIndex>(device_index, "CUDA device"));
  } catch (const c10::Error &error) {
    throw timeline::CollectionError(timeline::CollectionErrorCode::Transfer, producer,
                                    "xpool observation IPC device selection failed", std::nullopt,
                                    std::string{std::string_view{error.what_without_backtrace()}.substr(0, 256)});
  }
}

} // namespace

std::size_t ObservationPool::owned_device_bytes(std::size_t capacity) {
  if (capacity == 0 || capacity > std::numeric_limits<std::size_t>::max() / allocation_bytes)
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, 0,
                                    "xpool observation IPC capacity is zero or overflows Device demand");
  return capacity * allocation_bytes;
}

std::size_t ObservationPool::required_device_bytes(std::size_t capacity, bool owns_allocations) {
  const auto per_region = state_allocation_bytes + (owns_allocations ? allocation_bytes : 0);
  if (capacity == 0 || capacity > std::numeric_limits<std::size_t>::max() / per_region)
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, 0,
                                    "xpool observation state and IPC demand is zero or overflows");
  return capacity * per_region;
}

std::shared_ptr<ObservationPool> ObservationPool::create(std::shared_ptr<timeline::MemoryBudget> host,
                                                         std::shared_ptr<timeline::MemoryBudget> device,
                                                         timeline::BatchSource source, int device_index,
                                                         std::size_t capacity) {
  if (!host || !device || host == device || source.producer == 0 || source.clock_domain_id == 0 ||
      std::all_of(source.run_id.begin(), source.run_id.end(), [](auto byte) { return byte == std::byte{0}; }) ||
      device_index < 0 || device_index > std::numeric_limits<c10::DeviceIndex>::max())
    throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, source.producer,
                                    "xpool observation IPC requires shared budgets and a registered Device source");
  (void)owned_device_bytes(capacity);
  return timeline::make_budgeted_shared<ObservationPool>(host, timeline::MemoryKind::Control, source.producer,
                                                         ConstructionKey{}, host, std::move(device), source,
                                                         device_index, capacity);
}

ObservationPool::ObservationPool(ConstructionKey, std::shared_ptr<timeline::MemoryBudget> host,
                                 std::shared_ptr<timeline::MemoryBudget> device, timeline::BatchSource source,
                                 int device_index, std::size_t capacity)
    : host_(std::move(host)), device_(std::move(device)), source_(source), device_index_(device_index),
      resource_(*host_, timeline::MemoryKind::Control, source.producer), slots_(&resource_) {
  slots_.resize(capacity);
}

ObservationPool::Region::Region(ConstructionKey, std::shared_ptr<ObservationPool> pool, ObservationRegion descriptor,
                                bool owned)
    : pool_(std::move(pool)), descriptor_(descriptor), owned_(owned),
      charge_(pool_->device_->reserve(timeline::MemoryKind::Control, required_device_bytes(1, owned),
                                      pool_->source_.producer)) {}

void ObservationPool::Region::check(cudaError_t status, timeline::CollectionErrorCode code, const char *operation) {
  if (status != cudaSuccess)
    throw timeline::CollectionError(code, pool_->source_.producer, operation, std::nullopt, cudaGetErrorString(status));
}

void ObservationPool::Region::initialize() {
  const auto guard = guard_device(pool_->device_index_, pool_->source_.producer);
  if (owned_) {
    check(cudaMalloc(reinterpret_cast<void **>(&word_), allocation_bytes), timeline::CollectionErrorCode::Allocation,
          "xpool observation IPC allocation failed");
    // The synchronous cold copy finishes initialization before handle export.
    // A failed CUDA call can report earlier asynchronous failure; retain the
    // owner and charge even when no pointer was returned by that call.
    const auto initial = ObservationWord{};
    check(cudaMemcpy(word_, &initial, sizeof(initial), cudaMemcpyHostToDevice), timeline::CollectionErrorCode::Transfer,
          "xpool observation IPC initialization copy failed");
    check(cudaIpcGetMemHandle(&descriptor_.handle.value(), word_), timeline::CollectionErrorCode::Transfer,
          "xpool observation IPC handle export failed");
  } else {
    check(cudaIpcOpenMemHandle(reinterpret_cast<void **>(&word_), descriptor_.handle.value(),
                               cudaIpcMemLazyEnablePeerAccess),
          timeline::CollectionErrorCode::Transfer, "xpool observation IPC attachment failed");
  }
  check(cudaMalloc(reinterpret_cast<void **>(&operation_state_), state_allocation_bytes),
        timeline::CollectionErrorCode::Allocation, "xpool observation private state allocation failed");
  const auto initial_state = OperationState{};
  check(cudaMemcpy(operation_state_, &initial_state, sizeof(initial_state), cudaMemcpyHostToDevice),
        timeline::CollectionErrorCode::Transfer, "xpool observation private state initialization failed");
}

void ObservationPool::Region::release_storage() {
  const auto guard = guard_device(pool_->device_index_, pool_->source_.producer);
  check(owned_ ? cudaFree(word_) : cudaIpcCloseMemHandle(word_), timeline::CollectionErrorCode::Transfer,
        "xpool observation IPC retirement failed");
  word_ = nullptr;
  check(cudaFree(operation_state_), timeline::CollectionErrorCode::Transfer,
        "xpool observation private state retirement failed");
  operation_state_ = nullptr;
  charge_.reset();
  retired_.store(true, std::memory_order_release);
}

std::shared_ptr<ObservationPool::Region> ObservationPool::install(ObservationRegion descriptor, bool owned) {
  if (stopped_)
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, source_.producer,
                                    "xpool observation IPC admission is closed");
  const auto slot = std::find(slots_.begin(), slots_.end(), nullptr);
  if (slot == slots_.end())
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, source_.producer,
                                    "xpool observation IPC capacity exhausted");
  auto region =
      timeline::make_budgeted_shared<Region>(host_, timeline::MemoryKind::Control, source_.producer,
                                             Region::ConstructionKey{}, shared_from_this(), descriptor, owned);
  if (owned) {
    const auto sequence = endpoints_.try_next();
    if (sequence == 0) {
      stopped_ = true;
      identity_exhausted_ = true;
      throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, source_.producer,
                                      "xpool observation Endpoint identity exhausted");
    }
    region->descriptor_.endpoint_sequence = sequence;
  }
  // Mutual retention is installed in bounded, already admitted storage before
  // any CUDA operation. Only successful explicit retirement breaks this link.
  *slot = region;
  try {
    region->initialize();
  } catch (...) {
    region->uncertain_ = true;
    stopped_ = true;
    throw;
  }
  return region;
}

std::shared_ptr<ObservationPool::Region> ObservationPool::allocate(std::uint64_t generation) {
  if (generation == 0)
    throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, source_.producer,
                                    "xpool observation IPC allocation requires a Generation");
  const auto lock = std::lock_guard{mutex_};
  return install(ObservationRegion{.owner_producer = source_.producer,
                                   .generation = generation,
                                   .allocation_bytes = allocation_bytes},
                 true);
}

std::shared_ptr<ObservationPool::Region> ObservationPool::attach(const std::optional<ObservationRegion> &descriptor,
                                                                 bool transport_identity_required,
                                                                 std::uint64_t generation,
                                                                 std::uint64_t owner_producer) {
  validate_observation_region(descriptor, transport_identity_required, generation, owner_producer);
  if (!descriptor)
    return {};
  if (descriptor->allocation_bytes != allocation_bytes)
    throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, source_.producer,
                                    "xpool observation IPC extent differs from Native preparation");
  const auto lock = std::lock_guard{mutex_};
  for (const auto &entry : slots_)
    if (entry && (entry->descriptor_.handle == descriptor->handle ||
                  (entry->descriptor_.owner_producer == descriptor->owner_producer &&
                   entry->descriptor_.endpoint_sequence == descriptor->endpoint_sequence)))
      throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, source_.producer,
                                      "xpool observation IPC Endpoint or handle is already mapped");
  return install(*descriptor, false);
}

bool ObservationPool::retire(const std::shared_ptr<Region> &region, bool local_users_retired,
                             bool peer_mappings_detached) {
  const auto lock = std::lock_guard{mutex_};
  if (!region || region->pool_.get() != this)
    throw timeline::CollectionError(timeline::CollectionErrorCode::InvalidRelease, source_.producer,
                                    "xpool observation IPC retirement requires its original owner");
  if (region->retired())
    return true;
  if (!local_users_retired || (region->owned_ && !peer_mappings_detached) || region->uncertain_)
    return false;
  const auto slot = std::find(slots_.begin(), slots_.end(), region);
  if (slot == slots_.end())
    throw timeline::CollectionError(timeline::CollectionErrorCode::InvalidRelease, source_.producer,
                                    "xpool observation IPC retirement lost its resource slot");
  try {
    region->release_storage();
  } catch (...) {
    region->uncertain_ = true;
    stopped_ = true;
    throw;
  }
  slot->reset();
  return true;
}

void ObservationPool::stop() {
  const auto lock = std::lock_guard{mutex_};
  stopped_ = true;
}

ObservationPool::Report ObservationPool::report() const {
  const auto lock = std::lock_guard{mutex_};
  auto active = std::size_t{0};
  auto uncertain = std::size_t{0};
  for (const auto &entry : slots_) {
    active += entry != nullptr;
    uncertain += entry && entry->uncertain_;
  }
  return {slots_.size(), active, uncertain, endpoints_.quiescent_value(), stopped_, identity_exhausted_};
}

} // namespace xpool::transport
