#pragma once

/// \file xpool/devkit/timeline/resource.hpp
/// \brief Shared collection-memory admission and move-only resource charges.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <utility>

#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

/// Mutually exclusive physical allocation owners, independent of C0 wire data.
/// Classify each allocation once; never also charge its inclusive pool total.
enum class MemoryKind : std::size_t { Events, Control, Transfer, Delivery, Writer, Metadata, Snapshot, Threads, Count };

/// Consistent requested-storage accounting, excluding allocator/driver internals
/// and process RSS. All resident collection control must be charged explicitly.
struct MemoryReport {
  /// Shared byte ceiling installed before resource preparation.
  std::size_t limit = 0;
  /// Sum of currently retained physical-storage charges.
  std::size_t used = 0;
  /// Largest simultaneously charged byte total since construction.
  std::size_t peak = 0;
  /// Current mutually exclusive category charges, in bytes.
  std::array<std::size_t, static_cast<std::size_t>(MemoryKind::Count)> categories{};
  bool operator==(const MemoryReport &) const = default;
};

/// One process Host budget or one process/Device budget. This nonmoving owner
/// must outlive all borrowed reservations and allocations. Admission is a cold
/// Host operation, never an event-producer Hook. No storage is allocated here.
class MemoryBudget {
  class ConstructionKey {
    friend class MemoryBudget;
    ConstructionKey() = default;
  };

public:
  /// Create a shared ledger charging its actual allocator-requested object and
  /// shared control block before construction; weak references retain the charge.
  static std::shared_ptr<MemoryBudget> create(std::size_t limit);
  /// Allocator-only bootstrap; ordinary users call create() or the direct ctor.
  MemoryBudget(ConstructionKey, std::size_t limit, std::reference_wrapper<const std::size_t> resident);
  /// Charge declared resident control once, including this object if it is not
  /// already included in an enclosing owner allocation.
  /// \throws CollectionError resource_budget for zero/insufficient limits.
  explicit MemoryBudget(std::size_t limit, std::size_t resident_control_bytes = 0);
  /// A budget has one stable identity for every borrowed resource.
  MemoryBudget(const MemoryBudget &) = delete;
  /// Replacing a live accounting owner would invalidate its reservations.
  MemoryBudget &operator=(const MemoryBudget &) = delete;

  /// A charge for a resource managed elsewhere (pinned/Device memory or stacks).
  /// The actual owner proves retirement before reset, assignment or destruction.
  /// This value neither frees external memory nor establishes CUDA completion.
  class Reservation {
  public:
    /// Empty ownership, borrowing no budget.
    Reservation() = default;
    /// Return a charge whose externally owned resource has already retired.
    ~Reservation();
    /// Each charge has exactly one owner.
    Reservation(const Reservation &) = delete;
    /// Each charge has exactly one owner.
    Reservation &operator=(const Reservation &) = delete;
    /// Transfer ownership while emptying the source, without changing totals.
    Reservation(Reservation &&other) noexcept;
    /// Return the prior retired charge, then take the source's charge.
    Reservation &operator=(Reservation &&other) noexcept;
    /// Return the retired charge once; repeated calls are harmless.
    void reset() noexcept;
    /// Current owned bytes; zero for an empty or moved-from reservation.
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    friend class MemoryBudget;
    Reservation(MemoryBudget &budget, MemoryKind kind, std::size_t bytes) noexcept;
    MemoryBudget *budget_ = nullptr;
    MemoryKind kind_ = MemoryKind::Control;
    std::size_t bytes_ = 0;
  };

  /// An aligned pageable Host block and its inseparable budget charge.
  /// Move transfers both; reset/destruction frees the block before returning
  /// its charge. This is not pinned memory and has no Device-use retirement.
  class HostAllocation {
  public:
    /// Empty storage, borrowing no budget.
    HostAllocation() = default;
    /// Free storage before returning its charge.
    ~HostAllocation();
    /// Storage and its charge have one inseparable owner.
    HostAllocation(const HostAllocation &) = delete;
    /// Storage and its charge have one inseparable owner.
    HostAllocation &operator=(const HostAllocation &) = delete;
    /// Transfer pointer and charge together, leaving the source empty.
    HostAllocation(HostAllocation &&other) noexcept;
    /// Free the prior block before taking the source's pointer and charge.
    HostAllocation &operator=(HostAllocation &&other) noexcept;
    /// Free the block and then return its charge, once.
    void reset() noexcept;
    /// Borrow exactly the requested byte range until reset/destruction/move.
    std::span<std::byte> bytes() const noexcept { return {pointer_, bytes_}; }
    /// Actual rounded request charged to the budget.
    std::size_t charged_bytes() const noexcept { return reservation_.bytes(); }

  private:
    friend class MemoryBudget;
    HostAllocation(std::byte *pointer, std::size_t bytes, std::size_t alignment, Reservation reservation) noexcept;
    std::byte *pointer_ = nullptr;
    std::size_t bytes_ = 0;
    std::size_t alignment_ = 0;
    Reservation reservation_;
  };

  /// Charge before creating a resource. Zero bytes returns an empty reservation.
  /// Concurrent requests share the same limit. Failure preserves prior charges.
  /// \throws CollectionError resource_budget with the requesting Producer.
  Reservation reserve(MemoryKind kind, std::size_t bytes, std::uint64_t producer = 0);
  /// Admit rounded nonempty storage before aligned operator new. Allocation
  /// failure rolls its reservation back; all prior charges remain intact.
  /// \throws CollectionError resource_budget or allocation, retaining the cause.
  HostAllocation allocate_host(std::size_t bytes, std::size_t alignment, MemoryKind kind, std::uint64_t producer = 0);
  /// Allocator boundary for internal containers/control blocks. Pair with the
  /// exact original geometry after retirement; prefer HostAllocation otherwise.
  /// Zero-byte requests own no allocation. Charges precede allocation.
  void *allocate_storage(std::size_t bytes, std::size_t alignment, MemoryKind kind, std::uint64_t producer = 0);
  /// Free one allocator-owned retired block and return its charge. Matching
  /// pointer, size, alignment, category and a live budget are preconditions.
  void deallocate_storage(void *pointer, std::size_t bytes, std::size_t alignment, MemoryKind kind) noexcept;
  /// Snapshot used/peak and exclusive categories under the admission mutex.
  MemoryReport report() const;
  /// Round requested bytes using a positive power-of-two alignment, at least
  /// alignof(max_align_t). Zero bytes stays zero. No allocation is performed.
  /// \throws CollectionError resource_budget for invalid/overflowing geometry.
  static std::size_t rounded_bytes(std::size_t bytes, std::size_t alignment, std::uint64_t producer = 0);
  /// Checked count * declared element size, followed by the same Host rounding.
  /// Use actual sizeof/layout sizes, not guessed future component footprints.
  /// \throws CollectionError resource_budget for invalid/overflowing geometry.
  static std::size_t array_bytes(std::size_t count, std::size_t element_bytes, std::size_t alignment,
                                 std::uint64_t producer = 0);

private:
  void retire(MemoryKind kind, std::size_t bytes) noexcept;
  MemoryReport report_;
  mutable std::mutex mutex_;
};

/// PMR adapter for actual bounded internal metadata/registry containers. The
/// owner keeps the budget and resource alive through all allocations/containers.
class BudgetMemoryResource : public std::pmr::memory_resource {
public:
  /// Every allocation uses one exclusive category and diagnostic source.
  BudgetMemoryResource(MemoryBudget &budget, MemoryKind kind, std::uint64_t producer = 0)
      : budget_(budget), kind_(kind), producer_(producer) {}

private:
  void *do_allocate(std::size_t bytes, std::size_t alignment) override;
  void do_deallocate(void *pointer, std::size_t bytes, std::size_t alignment) override;
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;
  MemoryBudget &budget_;
  MemoryKind kind_;
  std::uint64_t producer_;
};

/// \cond INTERNAL
namespace detail {
template <typename T> class SharedBudgetAllocator {
public:
  using value_type = T;
  SharedBudgetAllocator(std::shared_ptr<MemoryBudget> budget, MemoryKind kind, std::uint64_t producer)
      : budget_(std::move(budget)), kind_(kind), producer_(producer) {}
  template <typename U>
  SharedBudgetAllocator(const SharedBudgetAllocator<U> &other) noexcept
      : budget_(other.budget_), kind_(other.kind_), producer_(other.producer_) {}
  T *allocate(std::size_t count) {
    if (count > static_cast<std::size_t>(-1) / sizeof(T))
      throw CollectionError(CollectionErrorCode::ResourceBudget, producer_, "xpool shared allocation size overflows");
    return static_cast<T *>(budget_->allocate_storage(count * sizeof(T), alignof(T), kind_, producer_));
  }
  void deallocate(T *pointer, std::size_t count) noexcept {
    budget_->deallocate_storage(pointer, count * sizeof(T), alignof(T), kind_);
  }
  template <typename U> bool operator==(const SharedBudgetAllocator<U> &other) const noexcept {
    return budget_ == other.budget_ && kind_ == other.kind_;
  }

private:
  template <typename U> friend class SharedBudgetAllocator;
  std::shared_ptr<MemoryBudget> budget_;
  MemoryKind kind_;
  std::uint64_t producer_;
};
} // namespace detail
/// \endcond

/// Charge a shared object's actual implementation-sized allocation, retaining
/// its budget through every strong and weak control-block reference.
/// \throws CollectionError on missing budget/admission/allocation failure.
template <typename T, typename... Args>
std::shared_ptr<T> make_budgeted_shared(std::shared_ptr<MemoryBudget> budget, MemoryKind kind, std::uint64_t producer,
                                        Args &&...args) {
  if (!budget)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool missing shared memory budget");
  return std::allocate_shared<T>(detail::SharedBudgetAllocator<T>{std::move(budget), kind, producer},
                                 std::forward<Args>(args)...);
}

} // namespace xpool::devkit::timeline
