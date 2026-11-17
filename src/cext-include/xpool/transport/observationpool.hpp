#pragma once

/// \file xpool/transport/observationpool.hpp
/// \brief Bounded, explicitly retired ownership of observation IPC storage.

#include <atomic>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <vector>

#include <xpool/arena.hpp>
#include <xpool/devkit/timeline/batch.hpp>
#include <xpool/devkit/timeline/counter.hpp>
#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/transport/observation.hpp>

namespace xpool::transport {

/// One registered process/Device Producer's bounded mapping owner. Preparation
/// must create this once per Producer and retain it across Arena recreation;
/// replacing it under the same Producer would reuse Endpoint sequences.
/// Cold lifecycle calls serialize under a Host mutex. No kernel uses this lock.
/// Live regions and the pool retain one another until explicit retirement,
/// including after a caller drops its reference or a CUDA result is uncertain.
class ObservationPool : public std::enable_shared_from_this<ObservationPool> {
  class ConstructionKey {
    friend class ObservationPool;
    ConstructionKey() = default;
  };

public:
  /// One allocation or import, retained in a fixed pool slot until retirement.
  /// Destruction alone never establishes that a CUDA resource may be released.
  class Region {
    class ConstructionKey {
      friend class ObservationPool;
      ConstructionKey() = default;
    };

  public:
    Region(const Region &) = delete;
    Region &operator=(const Region &) = delete;
    /// Immutable descriptor, available after successful allocation/attachment.
    const ObservationRegion &descriptor() const noexcept { return descriptor_; }
    /// Borrow the Device word until successful retirement. The caller serializes
    /// use against retirement; this grants no permission for live Host reads.
    ObservationWord *word() const noexcept { return word_; }
    /// Borrow this process's private Device counter until successful retirement.
    /// It is never exported through IPC; Host inspection requires quiescence.
    OperationState *operation_state() const noexcept { return operation_state_; }
    /// True only after successful CUDA release and charge return.
    bool retired() const noexcept { return retired_.load(std::memory_order_acquire); }
    /// Allocation owners additionally require actual peer detach before release.
    bool owns_allocation() const noexcept { return owned_; }
    /// Allocator-only construction; use the pool's allocation/attachment calls.
    Region(ConstructionKey, std::shared_ptr<ObservationPool> pool, ObservationRegion descriptor, bool owned);

  private:
    friend class ObservationPool;
    void initialize();
    void release_storage();
    void check(cudaError_t status, xpool::devkit::timeline::CollectionErrorCode code, const char *operation);
    const std::shared_ptr<ObservationPool> pool_;
    ObservationRegion descriptor_;
    const bool owned_;
    xpool::devkit::timeline::MemoryBudget::Reservation charge_;
    ObservationWord *word_ = nullptr;
    OperationState *operation_state_ = nullptr;
    bool uncertain_ = false;
    std::atomic<bool> retired_{false};
  };

  /// Cold inspection; live slots include resources retained after CUDA errors.
  struct Report {
    /// Fixed simultaneous mapping capacity.
    std::size_t capacity;
    /// Slots whose resources have not been proven released.
    std::size_t active;
    /// Slots quarantined after an uncertain CUDA result.
    std::size_t uncertain;
    /// Last issued Endpoint; failed CUDA attempts also consume their identities.
    std::uint64_t endpoint_sequence;
    /// Future allocation and attachment admission is closed.
    bool stopped;
    /// The positive Endpoint namespace exhausted without wrapping.
    bool identity_exhausted;
  };

  /// The public eight-byte word rounded to the existing CUDA allocation unit.
  static constexpr std::size_t allocation_bytes = xpool::arena::kAllocationAlignment;
  /// Private per-process counter allocation, rounded using the same arena unit.
  static constexpr std::size_t state_allocation_bytes =
      ((sizeof(OperationState) + xpool::arena::kAllocationAlignment - 1) / xpool::arena::kAllocationAlignment) *
      xpool::arena::kAllocationAlignment;
  /// Admit fixed Host control/slots without touching CUDA. Both ledgers are
  /// supplied by common preparation, never created privately by this pool.
  /// Both peers charge private Device state before CUDA; only the exporter
  /// charges the IPC extent. Importing the same IPC storage does not charge it twice.
  /// Complete startup must admit this demand alongside all other providers.
  /// \throws timeline::CollectionError for invalid facts or Host admission.
  static std::shared_ptr<ObservationPool> create(std::shared_ptr<xpool::devkit::timeline::MemoryBudget> host,
                                                 std::shared_ptr<xpool::devkit::timeline::MemoryBudget> device,
                                                 xpool::devkit::timeline::BatchSource source, int device_index,
                                                 std::size_t capacity);
  ObservationPool(const ObservationPool &) = delete;
  ObservationPool &operator=(const ObservationPool &) = delete;
  /// Demand for simultaneous owned extents, including alignment, counted once.
  /// Importers allocate no new Device extent for the same IPC storage.
  /// \throws timeline::CollectionError for zero capacity or size overflow.
  static std::size_t owned_device_bytes(std::size_t capacity);
  /// Complete pool demand: private states on both peers plus owned IPC extents
  /// on the exporter, each rounded allocation counted once. Other providers
  /// must be admitted alongside this demand by process preparation.
  /// \throws timeline::CollectionError for zero capacity or size overflow.
  static std::size_t required_device_bytes(std::size_t capacity, bool owns_allocations);
  /// Allocate on the owning Arena's cold creation path. Reserve a slot and both
  /// charges before CUDA, then issue a fresh Endpoint immediately before its
  /// allocation. CUDA failures retain the bounded slot/resources and close
  /// admission; their identities are never reused. No Event slot is involved.
  /// \throws timeline::CollectionError for invalid Generation/admission/CUDA.
  std::shared_ptr<Region> allocate(std::uint64_t generation);
  /// Validate policy, identities and exact extent before any CUDA call. Null
  /// with disabled policy returns empty. Live duplicate Endpoints or handles
  /// reject. The expected owner comes from actual remote preparation facts.
  /// \throws timeline::CollectionError for mismatch, admission or CUDA failure.
  std::shared_ptr<Region> attach(const std::optional<ObservationRegion> &descriptor, bool transport_identity_required,
                                 std::uint64_t generation, std::uint64_t owner_producer);
  /// Release on the retirement lane only after all local kernel/Graph/collector
  /// users retire, and (for owned extents) actual peer mappings detach. Missing
  /// proof or prior uncertainty returns false without any CUDA call. Successful
  /// retries are harmless. A timeout retains the lane, pool and all resources;
  /// this blocking CUDA operation does not itself implement a deadline.
  /// \throws timeline::CollectionError on foreign ownership or CUDA failure.
  bool retire(const std::shared_ptr<Region> &region, bool local_users_retired, bool peer_mappings_detached);
  /// Close new admission without releasing live or uncertain resources.
  void stop();
  /// Inspect under the cold lifecycle lock; no CUDA synchronization or reads.
  Report report() const;
  /// Immutable registered local Device Source, distinct from an imported owner.
  const xpool::devkit::timeline::BatchSource &source() const noexcept { return source_; }
  /// Actual process-local CUDA ordinal supplied by preparation.
  int device_index() const noexcept { return device_index_; }
  /// Borrow the common Host ledger for the containing Arena's retained control.
  const std::shared_ptr<xpool::devkit::timeline::MemoryBudget> &host_budget() const noexcept { return host_; }
  /// Borrow the same Device ledger when admitting Resident observation storage.
  const std::shared_ptr<xpool::devkit::timeline::MemoryBudget> &device_budget() const noexcept { return device_; }
  /// Budgeted Host allocator for arrays whose enclosing owner retains this pool.
  /// Allocations are charged until actual deallocation, including failed startup.
  std::pmr::memory_resource *host_resource() noexcept { return &resource_; }
  /// Allocator-only construction; create() admits the object and control block.
  ObservationPool(ConstructionKey, std::shared_ptr<xpool::devkit::timeline::MemoryBudget> host,
                  std::shared_ptr<xpool::devkit::timeline::MemoryBudget> device,
                  xpool::devkit::timeline::BatchSource source, int device_index, std::size_t capacity);

private:
  std::shared_ptr<Region> install(ObservationRegion descriptor, bool owned);
  const std::shared_ptr<xpool::devkit::timeline::MemoryBudget> host_;
  const std::shared_ptr<xpool::devkit::timeline::MemoryBudget> device_;
  const xpool::devkit::timeline::BatchSource source_;
  const int device_index_;
  xpool::devkit::timeline::BudgetMemoryResource resource_;
  std::pmr::vector<std::shared_ptr<Region>> slots_;
  xpool::devkit::timeline::AttemptSequence endpoints_;
  mutable std::mutex mutex_;
  bool stopped_ = false;
  bool identity_exhausted_ = false;
};

} // namespace xpool::transport
