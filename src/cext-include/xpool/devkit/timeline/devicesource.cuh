#pragma once

/// \file xpool/devkit/timeline/devicesource.cuh
/// \brief Device source identities, immutable publication and cumulative outcomes.

#include <cstddef>
#include <cstdint>

#include <cuda/atomic>
#include <cuda/std/span>

#include <xpool/devkit/timeline/device.cuh>
#include <xpool/devkit/timeline/devicesequence.cuh>
#include <xpool/devkit/timeline/source.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Concurrent Device source over fixed borrowed storage. Every call executes
/// on the same Device; at most kAttemptConcurrencyLimit reservations overlap.
/// Initialization and reclamation remain Host-owner lifecycle responsibilities.
class DeviceEventSource {
public:
  /// Borrow registered provenance, matching spans and initialized disjoint
  /// states. Storage outlives all calls and held reads. DeviceSourceState history
  /// is validated by the Host owner before launch, not on the production path.
  XPOOL_DEVICE_FN DeviceEventSource(BatchSource source, cuda::std::span<Event> events,
                                    cuda::std::span<SlotControl> slots, SlotPoolState &pool_state,
                                    DeviceSourceState &state)
      : source_(source), pool_(events, slots, pool_state), state_(state) {}

  /// Allocate one positive identity for Reserved/Full, with bounded atomics and
  /// one scan. Calls accepted before concurrent stop may finish; later entries
  /// allocate nothing. Failed results grant zero tokens. Invalid registration
  /// or geometry is a violated setup precondition and grants no ownership.
  XPOOL_DEVICE_FN ReserveResult try_reserve() const {
    if (source_.producer == 0 || source_.clock_domain_id == 0 || pool_.slots_.empty() ||
        pool_.slots_.size() != pool_.events_.size()) {
      return {.status = ReserveStatus::Stopped};
    }
    auto gate = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission};
    const auto previous = gate.fetch_add(std::uint64_t{1}, cuda::memory_order_acquire);
    if ((previous & (stopped | exhausted)) != 0) {
      gate.fetch_sub(std::uint64_t{1}, cuda::memory_order_release);
      return {.status = (previous & exhausted) != 0 ? ReserveStatus::IdentityExhausted : ReserveStatus::Stopped};
    }
    const auto sequence = next_identity();
    if (sequence == 0) {
      gate.fetch_or(stopped | exhausted, cuda::memory_order_relaxed);
      gate.fetch_sub(std::uint64_t{1}, cuda::memory_order_release);
      return {.status = ReserveStatus::IdentityExhausted};
    }
    const auto result = pool_.reserve_admitted(sequence);
    if (result.status == ReserveStatus::Full) {
      add(state_.dropped_full);
    }
    if (sequence == ~std::uint64_t{0}) {
      gate.fetch_or(stopped | exhausted, cuda::memory_order_relaxed);
    }
    gate.fetch_sub(std::uint64_t{1}, cuda::memory_order_release);
    return result;
  }

  /// Publish once and count the result. Event fields describe this registered
  /// source and its clock domain; the issued token supplies event_sequence.
  /// A collective Hook selects exactly one thread to own each Event write.
  XPOOL_DEVICE_FN CommitStatus commit(WriteToken token, const Event &event) const {
    const auto result = pool_.commit(token, event);
    if (result == CommitStatus::Committed) {
      add(state_.committed);
    } else if (result == CommitStatus::AlreadyCommitted) {
      add(state_.duplicate_commit);
    } else {
      add(state_.stale_token);
    }
    return result;
  }

  /// Retire an uncommitted reservation once. Rejected calls are diagnostics,
  /// rather than additional attempted events or successful abandons.
  XPOOL_DEVICE_FN AbandonStatus abandon(WriteToken token) const {
    const auto result = pool_.abandon(token);
    if (result == AbandonStatus::Abandoned) {
      add(state_.abandoned);
      retire_generation(token.slot_generation);
    } else {
      add(state_.stale_token);
    }
    return result;
  }

  /// Acquire an immutable committed value, skipping earlier pending identities.
  XPOOL_DEVICE_FN ReadResult try_acquire() const { return pool_.try_acquire(); }
  /// One candidate for a bounded collection scan, retaining normal ownership.
  XPOOL_DEVICE_FN ReadResult try_acquire_at(std::size_t index) const { return pool_.try_acquire_at(index); }

  /// Retire only after every read and asynchronous transfer has completed.
  /// Submission and timeout never establish that a held pointer may be reused.
  XPOOL_DEVICE_FN ReleaseStatus release(ReadToken token) const {
    const auto result = pool_.release(token);
    if (result == ReleaseStatus::Released) {
      retire_generation(token.slot_generation);
    } else {
      add(state_.stale_token);
    }
    return result;
  }

  /// Close new admission while existing write and copy owners remain valid.
  XPOOL_DEVICE_FN void stop() const {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.fetch_or(stopped,
                                                                                          cuda::memory_order_release);
  }

  /// Read admission without reserving an Event or advancing any counter.
  /// Accepted tokens remain valid even when later admission is closed.
  XPOOL_DEVICE_FN bool is_stopped() const {
    const auto gate =
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.load(cuda::memory_order_acquire);
    return (gate & (stopped | exhausted)) != 0;
  }

  /// Close this Source after a dependent observation identity exhausts. Existing
  /// reservations remain retireable; quality preserves the exhaustion reason.
  XPOOL_DEVICE_FN void stop_identity_exhausted() const {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.fetch_or(stopped | exhausted,
                                                                                          cuda::memory_order_release);
  }

  /// Sample cumulative Device atomic lower bounds during service. Pending is
  /// unknown; readers retain maxima and never project this as an exact final
  /// checkpoint. Host use still waits for the snapshot transfer to complete.
  XPOOL_DEVICE_FN SourceProgress progress() const {
    const auto gate =
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.load(cuda::memory_order_acquire);
    const auto upper = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.attempts.upper}.load(
        cuda::memory_order_relaxed);
    const auto lower = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.attempts.lower}.load(
        cuda::memory_order_relaxed);
    const auto attempted = upper != 0 ? lower_capacity + (upper < upper_capacity ? upper : upper_capacity)
                                      : (lower < lower_capacity ? lower : lower_capacity);
    const auto committed = counter(state_.committed);
    const auto full = counter(state_.dropped_full);
    const auto abandoned = counter(state_.abandoned);
    const auto stale = counter(state_.stale_token);
    const auto duplicate = counter(state_.duplicate_commit);
    auto result = SourceCounts{.attempted = attempted,
                               .last_attempted_sequence = attempted,
                               .committed = committed.value,
                               .dropped_full = full.value,
                               .abandoned = abandoned.value,
                               .stale_token = stale.value,
                               .duplicate_commit = duplicate.value,
                               .counter_exhausted = committed.exhausted || full.exhausted || abandoned.exhausted ||
                                                    stale.exhausted || duplicate.exhausted,
                               .stopped = (gate & stopped) != 0,
                               .identity_exhausted = (gate & exhausted) != 0};
    result.exact_totals = false;
    return {.counts = result, .pending_known = false};
  }

  /// Source-local totals after every API call has retired; Reserved/Copying
  /// owners may remain. This is not live C0 quality or a Host/Device handshake.
  /// Inconsistent observations and saturation clear exact_totals; they never
  /// claim a known complete final tail.
  XPOOL_DEVICE_FN SourceCounts quiescent_counts() const {
    auto result = progress().counts;
    const auto gate =
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.load(cuda::memory_order_acquire);
    auto stable = (gate & active_mask) == 0;
    for (auto &slot : pool_.slots_) {
      const auto word =
          cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership}.load(cuda::memory_order_acquire);
      const auto phase = DeviceSlotPool::phase_of(word);
      if (phase == DeviceSlotPool::Phase::Reserved) {
        ++result.pending;
      } else if (phase == DeviceSlotPool::Phase::Initializing || phase == DeviceSlotPool::Phase::Writing ||
                 phase == DeviceSlotPool::Phase::Recycling) {
        stable = false;
      }
    }
    auto remaining = result.attempted;
    const std::uint64_t terminal[] = {result.committed, result.dropped_full, result.abandoned, result.pending};
    for (const auto value : terminal) {
      if (value > remaining) {
        stable = false;
        break;
      }
      remaining -= value;
    }
    result.exact_totals = stable && remaining == 0 && !result.counter_exhausted;
    return result;
  }

  /// Immutable registration; Producers sharing a clock remain distinct sources.
  XPOOL_DEVICE_FN const BatchSource &source() const { return source_; }

  /// Host-only checked slot/source geometry using actual DeviceSourceState
  /// size and alignment. This query does not allocate or initialize CUDA.
  /// \throws std::invalid_argument for invalid dimensions/alignment and
  /// std::overflow_error for arithmetic overflow.
  static SourceLayout layout(std::size_t capacity, std::size_t allocation_alignment = 256) {
    return DeviceSourceState::layout(capacity, allocation_alignment);
  }

private:
  static constexpr std::uint64_t stopped = std::uint64_t{1} << 62;
  static constexpr std::uint64_t exhausted = std::uint64_t{1} << 63;
  static constexpr std::uint64_t active_mask = stopped - 1;
  static constexpr std::uint64_t upper_capacity = std::uint64_t{1} << 63;
  static constexpr std::uint64_t lower_capacity = upper_capacity - 1;

  XPOOL_DEVICE_FN std::uint64_t next_identity() const { return detail::next_device_sequence(state_.attempts); }

  XPOOL_DEVICE_FN static void add(DeviceCounterState &state) {
    if (cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.exhausted}.load(cuda::memory_order_relaxed) !=
        0) {
      return;
    }
    const auto previous = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.value}.fetch_add(
        std::uint64_t{1}, cuda::memory_order_relaxed);
    if (previous == ~std::uint64_t{0}) {
      cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.exhausted}.store(std::uint64_t{1},
                                                                                        cuda::memory_order_release);
    }
  }

  XPOOL_DEVICE_FN static CounterSnapshot counter(DeviceCounterState &state) {
    const auto overflow =
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.exhausted}.load(cuda::memory_order_acquire);
    return {.value = overflow != 0 ? ~std::uint64_t{0}
                                   : cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.value}.load(
                                         cuda::memory_order_relaxed),
            .exhausted = overflow != 0};
  }

  XPOOL_DEVICE_FN void retire_generation(std::uint64_t generation) const {
    if (generation == kSlotGenerationLimit) {
      cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.admission}.fetch_or(stopped | exhausted,
                                                                                            cuda::memory_order_release);
    }
  }

  BatchSource source_;
  DeviceSlotPool pool_;
  DeviceSourceState &state_;
};

} // namespace xpool::devkit::timeline
