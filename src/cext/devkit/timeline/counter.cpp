#include <xpool/devkit/timeline/counter.hpp>

#include <algorithm>

namespace xpool::devkit::timeline {

namespace {

constexpr auto upper_capacity = std::uint64_t{1} << 63;
constexpr auto lower_capacity = upper_capacity - 1;
static_assert(kAttemptConcurrencyLimit < lower_capacity);

} // namespace

AttemptSequence::AttemptSequence(std::uint64_t last_identity) noexcept
    : lower_(std::min(last_identity, lower_capacity)),
      upper_(last_identity > lower_capacity ? last_identity - lower_capacity : 0),
      upper_range_(last_identity >= lower_capacity ? 1 : 0) {}

std::uint64_t AttemptSequence::try_next() noexcept {
  auto range = std::atomic_ref{upper_range_};
  // RMW reads the latest modification, rather than permitting a new caller to
  // keep adding to an obsolete range after its permanent transition.
  if (range.fetch_or(std::uint64_t{0}, std::memory_order_acquire) == 0) {
    const auto ticket = std::atomic_ref{lower_}.fetch_add(std::uint64_t{1}, std::memory_order_relaxed);
    if (ticket < lower_capacity) {
      if (ticket == lower_capacity - 1) {
        range.store(std::uint64_t{1}, std::memory_order_release);
      }
      return ticket + 1;
    }
    range.store(std::uint64_t{1}, std::memory_order_release);
  }
  auto upper = std::atomic_ref{upper_};
  if (upper.fetch_or(std::uint64_t{0}, std::memory_order_relaxed) >= upper_capacity) {
    return 0;
  }
  const auto ticket = upper.fetch_add(std::uint64_t{1}, std::memory_order_relaxed);
  return ticket < upper_capacity ? upper_capacity + ticket : 0;
}

std::uint64_t AttemptSequence::lower_bound_value() const noexcept {
  const auto upper = std::atomic_ref{upper_}.load(std::memory_order_relaxed);
  return upper != 0 ? lower_capacity + std::min(upper, upper_capacity)
                    : std::min(std::atomic_ref{lower_}.load(std::memory_order_relaxed), lower_capacity);
}

std::uint64_t AttemptSequence::quiescent_value() const noexcept { return lower_bound_value(); }

} // namespace xpool::devkit::timeline
