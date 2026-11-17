#include <xpool/devkit/timeline/resource.hpp>

#include <algorithm>
#include <bit>
#include <new>
#include <string>
#include <utility>

#include <xpool/devkit/timeline/error.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {

namespace {
// allocate_shared requests its complete control block before invoking the
// ledger constructor. The bootstrap pointer is used only for that allocation;
// later deallocation never dereferences its expired stack observation.
template <typename T> struct BootstrapAllocator {
  using value_type = T;
  std::size_t *bytes;
  std::size_t limit;
  template <typename U>
  BootstrapAllocator(const BootstrapAllocator<U> &other) : bytes(other.bytes), limit(other.limit) {}
  BootstrapAllocator(std::size_t &observed, std::size_t maximum) : bytes(&observed), limit(maximum) {}
  T *allocate(std::size_t count) {
    const auto requested = xpool::utils::checked::prod(count, sizeof(T));
    if (requested > limit)
      throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool budget control exceeds its own limit");
    *bytes = requested;
    return static_cast<T *>(::operator new(requested, std::align_val_t{alignof(T)}));
  }
  void deallocate(T *pointer, std::size_t) noexcept { ::operator delete(pointer, std::align_val_t{alignof(T)}); }
  template <typename U> bool operator==(const BootstrapAllocator<U> &other) const noexcept {
    return bytes == other.bytes;
  }
};
} // namespace

std::shared_ptr<MemoryBudget> MemoryBudget::create(std::size_t limit) {
  auto resident = std::size_t{0};
  try {
    return std::allocate_shared<MemoryBudget>(BootstrapAllocator<MemoryBudget>{resident, limit}, ConstructionKey{},
                                              limit, std::cref(resident));
  } catch (const std::bad_alloc &cause) {
    throw CollectionError(CollectionErrorCode::Allocation, 0, "xpool budget allocation failed", std::nullopt,
                          cause.what());
  }
}

MemoryBudget::MemoryBudget(ConstructionKey, std::size_t limit, std::reference_wrapper<const std::size_t> resident)
    : MemoryBudget(limit, resident.get()) {}

MemoryBudget::MemoryBudget(std::size_t limit, std::size_t resident_control_bytes) {
  if (limit == 0 || resident_control_bytes > limit)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool invalid collection memory limit");
  report_.limit = limit;
  report_.used = resident_control_bytes;
  report_.peak = resident_control_bytes;
  report_.categories[static_cast<std::size_t>(MemoryKind::Control)] = resident_control_bytes;
}

MemoryBudget::Reservation MemoryBudget::reserve(MemoryKind kind, std::size_t bytes, std::uint64_t producer) {
  const auto index = static_cast<std::size_t>(kind);
  const auto lock = std::lock_guard{mutex_};
  if (index >= report_.categories.size() || bytes > report_.limit - report_.used) {
    const auto cause = "category=" + std::to_string(index) + " requested=" + std::to_string(bytes) +
                       " used=" + std::to_string(report_.used) + " limit=" + std::to_string(report_.limit);
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool collection memory admission failed",
                          std::nullopt, cause);
  }
  if (bytes == 0)
    return {};
  // The remaining-capacity comparison proves both additions fit size_t.
  report_.used += bytes;
  report_.categories[index] += bytes;
  report_.peak = std::max(report_.peak, report_.used);
  return Reservation{*this, kind, bytes};
}

void MemoryBudget::retire(MemoryKind kind, std::size_t bytes) noexcept {
  const auto lock = std::lock_guard{mutex_};
  // Only a live, private, move-only reservation can reach this operation.
  report_.categories[static_cast<std::size_t>(kind)] -= bytes;
  report_.used -= bytes;
}

MemoryReport MemoryBudget::report() const {
  const auto lock = std::lock_guard{mutex_};
  return report_;
}

MemoryBudget::Reservation::Reservation(MemoryBudget &budget, MemoryKind kind, std::size_t bytes) noexcept
    : budget_(&budget), kind_(kind), bytes_(bytes) {}
MemoryBudget::Reservation::~Reservation() { reset(); }
MemoryBudget::Reservation::Reservation(Reservation &&other) noexcept
    : budget_(std::exchange(other.budget_, nullptr)), kind_(other.kind_), bytes_(std::exchange(other.bytes_, 0)) {}
MemoryBudget::Reservation &MemoryBudget::Reservation::operator=(Reservation &&other) noexcept {
  if (this != &other) {
    reset();
    budget_ = std::exchange(other.budget_, nullptr);
    kind_ = other.kind_;
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}
void MemoryBudget::Reservation::reset() noexcept {
  if (budget_ != nullptr) {
    budget_->retire(kind_, bytes_);
    budget_ = nullptr;
    bytes_ = 0;
  }
}

std::size_t MemoryBudget::rounded_bytes(std::size_t bytes, std::size_t alignment, std::uint64_t producer) {
  if (!std::has_single_bit(alignment))
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool invalid host allocation alignment");
  try {
    return xpool::utils::checked::align_up(bytes, std::max(alignment, alignof(std::max_align_t)));
  } catch (const c10::Error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool host allocation rounding overflows",
                          std::nullopt, cause.what_without_backtrace());
  }
}

std::size_t MemoryBudget::array_bytes(std::size_t count, std::size_t element_bytes, std::size_t alignment,
                                      std::uint64_t producer) {
  if (count != 0 && element_bytes == 0)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool host element size must be positive");
  auto bytes = std::size_t{0};
  try {
    bytes = xpool::utils::checked::prod(count, element_bytes);
  } catch (const c10::Error &cause) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool host allocation size overflows",
                          std::nullopt, cause.what_without_backtrace());
  }
  return rounded_bytes(bytes, alignment, producer);
}

MemoryBudget::HostAllocation MemoryBudget::allocate_host(std::size_t bytes, std::size_t alignment, MemoryKind kind,
                                                         std::uint64_t producer) {
  if (bytes == 0)
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool host allocation must be nonempty");
  const auto rounded = rounded_bytes(bytes, alignment, producer);
  auto reservation = reserve(kind, rounded, producer);
  const auto effective_alignment = std::max(alignment, alignof(std::max_align_t));
  try {
    auto *pointer = static_cast<std::byte *>(::operator new(rounded, std::align_val_t{effective_alignment}));
    return HostAllocation{pointer, bytes, effective_alignment, std::move(reservation)};
  } catch (const std::bad_alloc &cause) {
    throw CollectionError(CollectionErrorCode::Allocation, producer, "xpool host collection allocation failed",
                          std::nullopt, cause.what());
  }
}

MemoryBudget::HostAllocation::HostAllocation(std::byte *pointer, std::size_t bytes, std::size_t alignment,
                                             Reservation reservation) noexcept
    : pointer_(pointer), bytes_(bytes), alignment_(alignment), reservation_(std::move(reservation)) {}
MemoryBudget::HostAllocation::~HostAllocation() { reset(); }
MemoryBudget::HostAllocation::HostAllocation(HostAllocation &&other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
      alignment_(std::exchange(other.alignment_, 0)), reservation_(std::move(other.reservation_)) {}
MemoryBudget::HostAllocation &MemoryBudget::HostAllocation::operator=(HostAllocation &&other) noexcept {
  if (this != &other) {
    reset();
    pointer_ = std::exchange(other.pointer_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    alignment_ = std::exchange(other.alignment_, 0);
    reservation_ = std::move(other.reservation_);
  }
  return *this;
}
void MemoryBudget::HostAllocation::reset() noexcept {
  if (pointer_ != nullptr) {
    ::operator delete(pointer_, std::align_val_t{alignment_});
    pointer_ = nullptr;
    bytes_ = 0;
    alignment_ = 0;
  }
  reservation_.reset();
}

void *MemoryBudget::allocate_storage(std::size_t bytes, std::size_t alignment, MemoryKind kind,
                                     std::uint64_t producer) {
  if (bytes == 0)
    return nullptr;
  auto block = allocate_host(bytes, alignment, kind, producer);
  auto *pointer = block.pointer_;
  block.pointer_ = nullptr;
  block.bytes_ = 0;
  block.alignment_ = 0;
  // Transfer accounting authority to the allocator's matching deallocation.
  block.reservation_.budget_ = nullptr;
  block.reservation_.bytes_ = 0;
  return pointer;
}

void MemoryBudget::deallocate_storage(void *pointer, std::size_t bytes, std::size_t alignment,
                                      MemoryKind kind) noexcept {
  if (pointer == nullptr)
    return;
  const auto effective = std::max(alignment, alignof(std::max_align_t));
  const auto rounded = bytes + (effective - bytes % effective) % effective;
  ::operator delete(pointer, std::align_val_t{effective});
  retire(kind, rounded);
}

void *BudgetMemoryResource::do_allocate(std::size_t bytes, std::size_t alignment) {
  return budget_.allocate_storage(std::max(bytes, std::size_t{1}), alignment, kind_, producer_);
}
void BudgetMemoryResource::do_deallocate(void *pointer, std::size_t bytes, std::size_t alignment) {
  budget_.deallocate_storage(pointer, std::max(bytes, std::size_t{1}), alignment, kind_);
}
bool BudgetMemoryResource::do_is_equal(const std::pmr::memory_resource &other) const noexcept { return this == &other; }

} // namespace xpool::devkit::timeline
