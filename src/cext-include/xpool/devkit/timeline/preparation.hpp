#pragma once

/// \file xpool/devkit/timeline/preparation.hpp
/// \brief Immutable process policy and common Host/disk preparation ledgers.

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

#include <xpool/devkit/timeline/disk.hpp>
#include <xpool/devkit/timeline/graphsession.hpp>
#include <xpool/devkit/timeline/policy.hpp>
#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/transport/observationpool.hpp>

namespace xpool::devkit::timeline {

/// Shared ledgers retained by metadata and the existing Graph compatibility
/// service. Borrowing them neither starts workers nor establishes retirement.
struct ProcessBudgets {
  /// All internal Host observation allocations use this one ledger.
  std::shared_ptr<MemoryBudget> host;
  /// Every runtime observation file shares the same process disk quota.
  std::shared_ptr<DiskBudget> disk;
  /// Common Device ledger once a Device provider is prepared; otherwise null.
  std::shared_ptr<MemoryBudget> device;
};

/// One process's cold preparation state. Configuration is installed before
/// Native initialization, and its fixed-size value is immutable thereafter.
/// Ledgers are allocated lazily for enabled observed roles. The Daemon uses
/// its separate launcher directory and never calls budgets(). No CUDA, worker
/// or file ownership is introduced here. Existing children keep their ledgers
/// alive even if an enclosing preparation owner retires.
class ProcessPreparation {
public:
  /// Return the single production owner shared with GraphLogSession.
  static ProcessPreparation &singleton();
  /// An unconfigured cold owner; construction allocates no collection storage.
  ProcessPreparation() = default;
  ProcessPreparation(const ProcessPreparation &) = delete;
  ProcessPreparation &operator=(const ProcessPreparation &) = delete;
  /// Freeze an already validated xpool.config projection. Equal retries succeed;
  /// changed policy, including its full configuration digest, is rejected.
  /// Existing Graph-only ledgers must have the same Host/disk limits.
  /// \throws CollectionError for policy mismatch or incomplete unified support.
  void configure(CollectionPolicy policy);
  /// Copy the installed value, or null before bootstrap configuration.
  std::optional<CollectionPolicy> read() const;
  /// Obtain common ledgers for an enabled, initialized observed process.
  /// The caller enforces its actual RuntimeRole. Disabled policy returns null
  /// and allocates nothing; missing policy fails explicitly.
  /// \throws CollectionError for missing configuration or resource admission.
  std::optional<ProcessBudgets> budgets();
  /// Tighten the observation retirement bound from the production owner's
  /// absolute CLOCK_MONOTONIC nanoseconds. Zero means already expired. Sampling
  /// steady_clock before CLOCK_MONOTONIC converts conservatively without assuming
  /// identical epochs. Later calls can shorten this bound, never renew it.
  /// No CUDA, workers or allocation are introduced.
  /// \throws CollectionError for a failed clock read or an unrepresentable bound.
  void limit_retirement(std::uint64_t monotonic_deadline_ns);
  /// Cap a caller's existing observation deadline by the known production bound.
  /// Without a production bound, preserve the supplied observation deadline.
  std::chrono::steady_clock::time_point
  retirement_deadline(std::chrono::steady_clock::time_point observation_deadline) const;
  /// Start or tighten the one observation close deadline, bounded by any
  /// production deadline. Every observer and retry shares this first close;
  /// expiration permits completion polling, never another waiting interval.
  /// \throws CollectionError for a wait outside zero through five seconds.
  std::chrono::steady_clock::time_point begin_retirement(std::chrono::milliseconds maximum_wait);
  /// Borrow the unique process service on the installed common budgets. The
  /// weak process reference adds no lifetime beyond actual owners/workers.
  /// Creates fixed Host/Device source slots and bounded exporter slots, without
  /// starting threads or CUDA. A live service is reused; an unjoined failure
  /// cannot be replaced. Graph installation precedes raw startup in SGLang.
  std::shared_ptr<CollectionService> collection_service();
  /// Prepare the same unique Device ledger for a Fabric participant, including
  /// FfnAgents without Transport IPC. No CUDA or worker is initialized here.
  /// Process-resident Host binding control and Device constant binding retain
  /// their charges until process retirement, even after the active view clears.
  ProcessBudgets fabric_budgets(std::size_t complete_device_bytes, std::uint64_t producer,
                                std::size_t resident_host_bytes, std::size_t resident_device_bytes);
  /// Prepare one pool for the actual local Device Producer, without CUDA.
  /// Disabled shared identity returns null. Equal retries reuse the pool;
  /// changed Source, ordinal or allocation ownership reject. Demand includes
  /// every slot's private state, exporter-owned IPC and the exporter's Resident
  /// tables/drain state, including each actual allocation's alignment.
  /// Future providers must use this same ledger and admit their complete demand
  /// alongside IPC before starting collection.
  /// \throws CollectionError for mismatched preparation or insufficient budget.
  std::shared_ptr<xpool::transport::ObservationPool> prepare_transport(BatchSource source, int device_index,
                                                                       bool owns_allocations);

private:
  friend class GraphLogSession;
  void require_graph_policy(const Options &options) const;
  ProcessBudgets acquire(std::size_t host_bytes, std::size_t disk_bytes, std::size_t metadata_reserve_bytes);
  std::shared_ptr<CollectionService> acquire_service(std::shared_ptr<MemoryBudget> host,
                                                     std::shared_ptr<DiskBudget> disk,
                                                     std::chrono::milliseconds cadence, std::size_t exporters);
  void require_limits(std::size_t host_bytes, std::size_t disk_bytes, std::size_t metadata_reserve_bytes) const;
  mutable std::mutex mutex_;
  std::optional<CollectionPolicy> policy_;
  std::optional<std::chrono::steady_clock::time_point> retirement_deadline_;
  std::weak_ptr<CollectionService> service_;
  std::shared_ptr<MemoryBudget> host_;
  std::shared_ptr<DiskBudget> disk_;
  std::shared_ptr<MemoryBudget> device_;
  std::shared_ptr<xpool::transport::ObservationPool> transport_;
  bool transport_owns_allocations_ = false;
  MemoryBudget::Reservation resident_;
  // Retained cold Python capacity: bounded identity DTOs, simultaneous decode/
  // validation/retry copies and binding temporaries. Native metadata allocations
  // are charged separately. Kept until process retirement, never on file export.
  MemoryBudget::Reservation python_preparation_;
  MemoryBudget::Reservation fabric_host_resident_;
  MemoryBudget::Reservation fabric_device_resident_;
};

} // namespace xpool::devkit::timeline
