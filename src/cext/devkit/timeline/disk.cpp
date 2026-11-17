#include <xpool/devkit/timeline/disk.hpp>

#include <algorithm>
#include <utility>

namespace xpool::devkit::timeline {

DiskBudget::DiskBudget(std::uint64_t limit, std::uint64_t metadata_reserve)
    : limit_(limit), metadata_reserve_(metadata_reserve) {
  if (limit == 0 || metadata_reserve == 0 || metadata_reserve >= limit) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool disk limits are invalid");
  }
}

DiskBudget::Extent::Extent(Extent &&other) noexcept
    : budget_(std::exchange(other.budget_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
DiskBudget::Extent &DiskBudget::Extent::operator=(Extent &&other) noexcept {
  if (this != &other) {
    budget_ = std::exchange(other.budget_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}
bool DiskBudget::Extent::replaced_by(Extent &&replacement) noexcept {
  if (&replacement == this || replacement.budget_ == nullptr || (budget_ != nullptr && budget_ != replacement.budget_))
    return false;
  if (budget_ != nullptr) {
    const auto lock = std::lock_guard{budget_->mutex_};
    budget_->used_ -= bytes_;
  }
  *this = std::move(replacement);
  return true;
}

DiskBudget::Reservation DiskBudget::reserve(std::uint64_t bytes, DiskKind kind, std::uint64_t producer) {
  const auto lock = std::lock_guard{mutex_};
  const auto ceiling = kind == DiskKind::Data ? limit_ - metadata_reserve_ : limit_;
  if (used_ > ceiling || reserved_ > ceiling - used_ || bytes > ceiling - used_ - reserved_) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer, "xpool process disk quota exceeded");
  }
  reserved_ += bytes;
  peak_ = std::max(peak_, used_ + reserved_);
  return {*this, bytes};
}

DiskBudget::Reservation::Reservation(DiskBudget &budget, std::uint64_t bytes) noexcept
    : budget_(&budget), bytes_(bytes) {}
DiskBudget::Reservation::~Reservation() { cancel(); }
DiskBudget::Reservation::Reservation(Reservation &&other) noexcept
    : budget_(std::exchange(other.budget_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
DiskBudget::Reservation &DiskBudget::Reservation::operator=(Reservation &&other) noexcept {
  if (this != &other) {
    cancel();
    budget_ = std::exchange(other.budget_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}
void DiskBudget::Reservation::cancel() noexcept {
  if (budget_ != nullptr) {
    const auto lock = std::lock_guard{budget_->mutex_};
    budget_->reserved_ -= bytes_;
    budget_ = nullptr;
    bytes_ = 0;
  }
}
void DiskBudget::Reservation::retain(std::uint64_t bytes) {
  if (budget_ == nullptr || bytes > bytes_) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool retained file extent exceeds admission");
  }
  const auto lock = std::lock_guard{budget_->mutex_};
  budget_->reserved_ -= bytes_;
  budget_->used_ += bytes;
  budget_ = nullptr;
  bytes_ = 0;
}

DiskBudget::Extent DiskBudget::Reservation::retain_extent(std::uint64_t bytes) {
  auto *budget = budget_;
  retain(bytes);
  return {*budget, bytes};
}

void DiskBudget::charge_existing(std::uint64_t bytes) {
  auto reservation = reserve(bytes, DiskKind::Metadata);
  reservation.retain(bytes);
}

DiskReport DiskBudget::report() const {
  const auto lock = std::lock_guard{mutex_};
  return {limit_, metadata_reserve_, used_, reserved_, peak_};
}

} // namespace xpool::devkit::timeline
