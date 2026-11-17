#pragma once

/// \file xpool/devkit/timeline/counter.hpp
/// \brief Full-width concurrent identities, outcomes and quiescent saturation.

#include <atomic>
#include <cstdint>
#include <limits>

namespace xpool::devkit::timeline {

/// Maximum overlapping identity allocations; private counter overshoot stays
/// below the unused half of each u64 word. This exceeds native process threads.
inline constexpr std::uint64_t kAttemptConcurrencyLimit = std::numeric_limits<std::uint32_t>::max();

/// Bounded concurrent allocation of every positive u64 identity exactly once.
/// Two disjoint ticket ranges leave room for in-flight callers at a boundary.
/// At most kAttemptConcurrencyLimit calls may overlap; storage remains alive
/// and is never reset during service. Snapshots require allocation quiescence.
class alignas(8) AttemptSequence {
public:
  /// Start after a retained identity, or zero for a fresh source. Initialization
  /// precedes all concurrent access; retained history is not crash recovery.
  explicit AttemptSequence(std::uint64_t last_identity = 0) noexcept;
  AttemptSequence(const AttemptSequence &) = delete;
  AttemptSequence &operator=(const AttemptSequence &) = delete;

  /// Allocate with a fixed number of lock-free atomic operations and no retry.
  /// Return zero only after the positive u64 range has been exhausted.
  std::uint64_t try_next() noexcept;
  /// Sample a cumulative lower bound during allocation. It is exact only after
  /// writers retire; readers may retain a maximum across samples.
  std::uint64_t lower_bound_value() const noexcept;
  /// Exact last allocated identity after all allocation calls retire. Private
  /// overshoot at either range boundary never appears as extra attempts.
  std::uint64_t quiescent_value() const noexcept;

private:
  static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
  alignas(std::atomic_ref<std::uint64_t>::required_alignment) mutable std::uint64_t lower_;
  alignas(std::atomic_ref<std::uint64_t>::required_alignment) mutable std::uint64_t upper_;
  alignas(std::atomic_ref<std::uint64_t>::required_alignment) std::uint64_t upper_range_;
};

/// Cumulative lower-bound evidence after overflow. One serialized owner updates
/// a counter; readers synchronize with that owner before inspecting its value.
class SaturatingCounter {
public:
  /// Add a nonnegative count; overflow saturates and remains explicitly marked.
  void add(std::uint64_t count) noexcept {
    if (count > std::numeric_limits<std::uint64_t>::max() - value_) {
      value_ = std::numeric_limits<std::uint64_t>::max();
      exhausted_ = true;
    } else {
      value_ += count;
    }
  }

  /// Exact cumulative value until overflow, then a preserved lower bound.
  std::uint64_t value() const noexcept { return value_; }
  /// Whether any addition exceeded the representable range.
  bool exhausted() const noexcept { return exhausted_; }

private:
  std::uint64_t value_ = 0;
  bool exhausted_ = false;
};

/// Logical concurrent counter value after all writing calls retire.
struct CounterSnapshot {
  /// Exact sum until overflow, then UINT64_MAX as lower-bound evidence.
  std::uint64_t value;
  /// Whether any update exceeded the full-width logical range.
  bool exhausted;
};

/// Bounded concurrent additions represented by a modulo word and a sticky
/// overflow flag. Quiescent readers clamp overflowed sums to UINT64_MAX;
/// intermediate words are not a live checkpoint. Retained initial values are
/// supplied only before concurrent access begins. Storage is never reset.
class alignas(8) ConcurrentCounter {
public:
  /// Start a fresh zero count before concurrent service.
  ConcurrentCounter() noexcept = default;
  /// Initialize a retained exact count before concurrent service.
  explicit ConcurrentCounter(std::uint64_t initial_value) noexcept : value_(initial_value) {}
  ConcurrentCounter(const ConcurrentCounter &) = delete;
  ConcurrentCounter &operator=(const ConcurrentCounter &) = delete;

  /// Add using a bounded number of lock-free atomic operations. Overflow marks
  /// the logical value saturated; every writer retires before a snapshot read.
  void add(std::uint64_t count) noexcept {
    if (std::atomic_ref{exhausted_}.load(std::memory_order_relaxed) != 0) {
      return;
    }
    const auto previous = std::atomic_ref{value_}.fetch_add(count, std::memory_order_relaxed);
    if (count > std::numeric_limits<std::uint64_t>::max() - previous) {
      std::atomic_ref{exhausted_}.store(std::uint64_t{1}, std::memory_order_release);
    }
  }

  /// Sample lower-bound evidence while calls may still run. A preempted
  /// overflow can temporarily expose a smaller private value; live projection
  /// retains its previous maximum and never claims exact totals.
  CounterSnapshot lower_bound() const noexcept {
    const auto exhausted = std::atomic_ref{exhausted_}.load(std::memory_order_acquire) != 0;
    return {.value = exhausted ? std::numeric_limits<std::uint64_t>::max()
                               : std::atomic_ref{value_}.load(std::memory_order_relaxed),
            .exhausted = exhausted};
  }

  /// Inspect after every writer quiesces for the exact or saturated total.
  CounterSnapshot quiescent() const noexcept { return lower_bound(); }

private:
  static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
  alignas(std::atomic_ref<std::uint64_t>::required_alignment) mutable std::uint64_t value_ = 0;
  alignas(std::atomic_ref<std::uint64_t>::required_alignment) mutable std::uint64_t exhausted_ = 0;
};

} // namespace xpool::devkit::timeline
