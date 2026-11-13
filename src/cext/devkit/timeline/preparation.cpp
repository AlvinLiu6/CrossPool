#include <xpool/devkit/timeline/preparation.hpp>

#include <algorithm>
#include <ctime>
#include <utility>

#include <xpool/transport/atnagent.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {

ProcessPreparation &ProcessPreparation::singleton() {
  static auto preparation = ProcessPreparation{};
  return preparation;
}

void ProcessPreparation::limit_retirement(std::uint64_t monotonic_deadline_ns) {
  const auto before = std::chrono::steady_clock::now();
  auto sampled = timespec{};
  if (::clock_gettime(CLOCK_MONOTONIC, &sampled) != 0)
    throw CollectionError(CollectionErrorCode::Io, 0, "xpool production deadline clock read failed");
  const auto now =
      static_cast<std::uint64_t>(sampled.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(sampled.tv_nsec);
  const auto remaining = monotonic_deadline_ns > now ? monotonic_deadline_ns - now : 0;
  const auto available =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::time_point::max() - before)
          .count();
  if (remaining > static_cast<std::uint64_t>(available))
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool production deadline exceeds clock range");
  const auto deadline = before + std::chrono::nanoseconds{static_cast<std::int64_t>(remaining)};
  const auto lock = std::lock_guard{mutex_};
  retirement_deadline_ = retirement_deadline_ ? std::min(*retirement_deadline_, deadline) : deadline;
}

std::chrono::steady_clock::time_point
ProcessPreparation::retirement_deadline(std::chrono::steady_clock::time_point observation_deadline) const {
  const auto lock = std::lock_guard{mutex_};
  return retirement_deadline_ ? std::min(*retirement_deadline_, observation_deadline) : observation_deadline;
}

std::chrono::steady_clock::time_point ProcessPreparation::begin_retirement(std::chrono::milliseconds maximum_wait) {
  if (maximum_wait.count() < 0 || maximum_wait.count() > 5000)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0,
                          "xpool shutdown wait requires zero through five seconds");
  const auto proposed = std::chrono::steady_clock::now() + maximum_wait;
  const auto lock = std::lock_guard{mutex_};
  retirement_deadline_ = retirement_deadline_ ? std::min(*retirement_deadline_, proposed) : proposed;
  return *retirement_deadline_;
}

std::shared_ptr<CollectionService> ProcessPreparation::acquire_service(std::shared_ptr<MemoryBudget> host,
                                                                       std::shared_ptr<DiskBudget> disk,
                                                                       std::chrono::milliseconds cadence,
                                                                       std::size_t exporters) {
  if (auto service = service_.lock(); service && !service->joined()) {
    if (service->host_budget() != host || service->disk_budget() != disk || service->cadence_ != cadence ||
        service->export_capacity_ != exporters)
      throw CollectionError(CollectionErrorCode::ResourceBudget, 0,
                            "xpool shared service preparation changed its geometry");
    return service;
  }
  auto service = CollectionService::create(std::move(host), std::move(disk), 2, cadence, exporters);
  service_ = service;
  return service;
}

std::shared_ptr<CollectionService> ProcessPreparation::collection_service() {
  const auto lock = std::lock_guard{mutex_};
  if (!policy_ || !policy_->enabled())
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool shared service requires enabled process policy");
  const auto budgets =
      acquire(policy_->limits.host_bytes, policy_->limits.disk_bytes, policy_->limits.metadata_reserve_bytes);
  return acquire_service(
      budgets.host, budgets.disk, std::chrono::milliseconds{policy_->limits.collection_interval_ms},
      xpool::utils::checked::sum(static_cast<std::size_t>(policy_->limits.endpoint_capacity), std::size_t{1}));
}

void ProcessPreparation::require_limits(std::size_t host_bytes, std::size_t disk_bytes,
                                        std::size_t metadata_reserve_bytes) const {
  if (host_ && (host_->report().limit != host_bytes || disk_->report().limit != disk_bytes ||
                disk_->report().metadata_reserve != metadata_reserve_bytes))
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0,
                          "xpool preparation limits differ from the process Host/disk ledgers");
}

void ProcessPreparation::configure(CollectionPolicy policy) {
  const auto lock = std::lock_guard{mutex_};
  if (policy.unified)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool unified preparation requires its production providers");
  if (policy_ && *policy_ != policy)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool observation policy cannot change after bootstrap");
  require_limits(policy.limits.host_bytes, policy.limits.disk_bytes, policy.limits.metadata_reserve_bytes);
  policy_ = policy;
}

std::optional<CollectionPolicy> ProcessPreparation::read() const {
  const auto lock = std::lock_guard{mutex_};
  return policy_;
}

ProcessBudgets ProcessPreparation::acquire(std::size_t host_bytes, std::size_t disk_bytes,
                                           std::size_t metadata_reserve_bytes) {
  require_limits(host_bytes, disk_bytes, metadata_reserve_bytes);
  if (!host_) {
    auto host = MemoryBudget::create(host_bytes);
    auto resident = host->reserve(MemoryKind::Control, sizeof(*this));
    // ObservationPreparation has at most 128 fixed-width participant and PE
    // entries. Reserve their Python construction/retention capacity before
    // bootstrap exposes any request or receives a Plan. This is conservative
    // admission capacity, not a claim about interpreter RSS or exact occupancy.
    auto python = host->reserve(MemoryKind::Metadata, 3 * 1024 * 1024);
    auto disk = make_budgeted_shared<DiskBudget>(host, MemoryKind::Control, 0, disk_bytes, metadata_reserve_bytes);
    host_ = std::move(host);
    disk_ = std::move(disk);
    resident_ = std::move(resident);
    python_preparation_ = std::move(python);
  }
  return {host_, disk_, device_};
}

std::shared_ptr<xpool::transport::ObservationPool>
ProcessPreparation::prepare_transport(BatchSource source, int device_index, bool owns_allocations) {
  const auto lock = std::lock_guard{mutex_};
  if (!policy_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool transport observation requires bootstrap policy");
  if (!policy_->transport_identity_required())
    return {};
  if (transport_) {
    if (transport_->source().run_id != source.run_id || transport_->source().producer != source.producer ||
        transport_->source().clock_domain_id != source.clock_domain_id || transport_->device_index() != device_index ||
        transport_owns_allocations_ != owns_allocations)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool transport observation cannot replace a prepared Device Producer");
    return transport_;
  }
  auto required =
      xpool::transport::ObservationPool::required_device_bytes(policy_->limits.endpoint_capacity, owns_allocations);
  if (owns_allocations) {
    try {
      required = xpool::utils::checked::sum(
          required, xpool::transport::resident_observation_device_bytes(policy_->limits.endpoint_capacity));
    } catch (const c10::Error &) {
      throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                            "xpool combined observation Region/Resident demand overflows");
    }
  }
  if (required > policy_->limits.device_bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool aligned observation state, IPC and Resident demand exceeds Device budget");
  (void)acquire(policy_->limits.host_bytes, policy_->limits.disk_bytes, policy_->limits.metadata_reserve_bytes);
  if (!device_)
    device_ =
        make_budgeted_shared<MemoryBudget>(host_, MemoryKind::Control, source.producer, policy_->limits.device_bytes);
  auto pool = xpool::transport::ObservationPool::create(host_, device_, source, device_index,
                                                        policy_->limits.endpoint_capacity);
  transport_owns_allocations_ = owns_allocations;
  transport_ = std::move(pool);
  return transport_;
}

std::optional<ProcessBudgets> ProcessPreparation::budgets() {
  const auto lock = std::lock_guard{mutex_};
  if (!policy_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool observation resources require bootstrap policy");
  if (!policy_->enabled())
    return std::nullopt;
  return acquire(policy_->limits.host_bytes, policy_->limits.disk_bytes, policy_->limits.metadata_reserve_bytes);
}

ProcessBudgets ProcessPreparation::fabric_budgets(std::size_t complete_device_bytes, std::uint64_t producer,
                                                  std::size_t resident_host_bytes, std::size_t resident_device_bytes) {
  const auto lock = std::lock_guard{mutex_};
  if (!policy_ || !policy_->fabric_events())
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, producer,
                          "xpool Fabric observation policy is unavailable");
  auto required = complete_device_bytes;
  if (transport_) {
    const auto ipc = xpool::transport::ObservationPool::required_device_bytes(policy_->limits.endpoint_capacity,
                                                                              transport_owns_allocations_);
    const auto resident = transport_owns_allocations_
                              ? xpool::transport::resident_observation_device_bytes(policy_->limits.endpoint_capacity)
                              : 0;
    try {
      required = xpool::utils::checked::sum(required, ipc, static_cast<std::size_t>(resident));
    } catch (const c10::Error &) {
      throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                            "xpool complete Fabric/IPC Device geometry overflows");
    }
  }
  if (required > policy_->limits.device_bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool complete Fabric/IPC observation demand exceeds Device budget");
  (void)acquire(policy_->limits.host_bytes, policy_->limits.disk_bytes, policy_->limits.metadata_reserve_bytes);
  if (!device_)
    device_ = make_budgeted_shared<MemoryBudget>(host_, MemoryKind::Control, producer, policy_->limits.device_bytes);
  if (fabric_host_resident_.bytes() == 0 && fabric_device_resident_.bytes() == 0) {
    auto host_charge = host_->reserve(MemoryKind::Control, resident_host_bytes, producer);
    auto device_charge = device_->reserve(MemoryKind::Control, resident_device_bytes, producer);
    fabric_host_resident_ = std::move(host_charge);
    fabric_device_resident_ = std::move(device_charge);
  } else if (fabric_host_resident_.bytes() != resident_host_bytes ||
             fabric_device_resident_.bytes() != resident_device_bytes) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool Fabric process-resident binding geometry cannot change");
  }
  return {host_, disk_, device_};
}

void ProcessPreparation::require_graph_policy(const Options &options) const {
  options.validate();
  if (options.enable)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool unified Graph preparation requires its production providers");
  if (policy_ && (!policy_->graph_events() || policy_->limits.host_bytes != options.host_bytes ||
                  policy_->limits.disk_bytes != options.disk_bytes ||
                  policy_->limits.metadata_reserve_bytes != options.metadata_reserve_bytes ||
                  policy_->limits.collection_interval_ms != options.collection_interval_ms ||
                  policy_->limits.shutdown_wait_ms != options.shutdown_wait_ms))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool Graph log preparation disagrees with bootstrap policy");
  require_limits(options.host_bytes, options.disk_bytes, options.metadata_reserve_bytes);
}

} // namespace xpool::devkit::timeline
