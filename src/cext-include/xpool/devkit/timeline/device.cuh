#pragma once

/// \file xpool/devkit/timeline/device.cuh
/// \brief Device-scoped immutable slots and completion-owned reclamation.

#include <cstddef>
#include <cstdint>

#include <cuda/atomic>
#include <cuda/std/span>

#include <xpool/devkit/timeline/buffer.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Borrowed Device storage using the Host slot layout and token types. Every
/// API operation runs on one Device; Host access requires explicit retirement
/// and synchronization. This is not a mapped Host/Device atomic protocol.
/// Each Event has one writing thread; different CTAs/Streams may own different
/// slots. The owner validates geometry and initializes controls before launch.
class DeviceSlotPool {
public:
  /// Borrow disjoint, equally sized payload/control spans and initialized
  /// scheduling state. Storage stays alive until every caller, copy and reader
  /// retires. Checked sizing is HostSlotPool::layout; source identity allocation
  /// and cumulative quality belong to the upper source layer.
  XPOOL_DEVICE_FN DeviceSlotPool(cuda::std::span<Event> events, cuda::std::span<SlotControl> slots,
                                 SlotPoolState &state)
      : events_(events), slots_(slots), state_(state) {}

  /// Supply a distinct positive u64 identity for every attempt, including Full.
  /// Scan each candidate once. Zero/invalid geometry grants no ownership and
  /// returns Stopped; valid callers establish these preconditions at setup.
  XPOOL_DEVICE_FN ReserveResult try_reserve(std::uint64_t sequence) const {
    if (sequence == 0 || slots_.empty() || slots_.size() != events_.size()) {
      return {.status = ReserveStatus::Stopped};
    }
    const auto flags =
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.stopped}.load(cuda::memory_order_acquire);
    if (flags != 0) {
      return {.status =
                  (flags & generation_exhausted) != 0 ? ReserveStatus::IdentityExhausted : ReserveStatus::Stopped};
    }
    return reserve_admitted(sequence);
  }

private:
  friend class DeviceEventSource;

  // The source gate has already admitted this attempt before concurrent stop.
  XPOOL_DEVICE_FN ReserveResult reserve_admitted(std::uint64_t sequence) const {
    auto index = static_cast<std::size_t>((sequence - 1) % slots_.size());
    for (auto inspected = std::size_t{0}; inspected < slots_.size(); ++inspected) {
      auto &slot = slots_[index];
      const auto previous = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership}.fetch_or(
          std::uint64_t{1}, cuda::memory_order_acquire);
      if (phase_of(previous) == Phase::Free) {
        const auto generation = previous >> 4;
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.event_sequence}.store(
            sequence, cuda::memory_order_relaxed);
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership}.store(
            ownership(generation, Phase::Reserved), cuda::memory_order_release);
        return {.status = ReserveStatus::Reserved,
                .token = {.slot = index, .slot_generation = generation, .event_sequence = sequence}};
      }
      index = index + 1 == slots_.size() ? 0 : index + 1;
    }
    return {.status = ReserveStatus::Full};
  }

public:
  /// Claim one generation before copying the complete Event and release-publish
  /// its completion. The issued token supplies event_sequence. Input stays
  /// immutable until this thread finishes the copy; no CTA barrier is implied.
  XPOOL_DEVICE_FN CommitStatus commit(WriteToken token, const Event &event) const {
    if (!matches(token)) {
      return CommitStatus::StaleToken;
    }
    auto expected = ownership(token.slot_generation, Phase::Reserved);
    auto &slot = slots_[token.slot];
    if (!cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership}.compare_exchange_strong(
            expected, ownership(token.slot_generation, Phase::Writing), cuda::memory_order_acq_rel,
            cuda::memory_order_acquire)) {
      return committed(expected, token.slot_generation) ? CommitStatus::AlreadyCommitted : CommitStatus::StaleToken;
    }
    events_[token.slot] = event;
    events_[token.slot].event_sequence = token.event_sequence;
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership}.store(
        ownership(token.slot_generation, Phase::Committed), cuda::memory_order_release);
    return CommitStatus::Committed;
  }

  /// Retire an uncommitted reservation once. A generation that has claimed
  /// commit remains immutable and cannot be abandoned or overwritten.
  XPOOL_DEVICE_FN AbandonStatus abandon(WriteToken token) const {
    if (!matches(token)) {
      return AbandonStatus::StaleToken;
    }
    auto expected = ownership(token.slot_generation, Phase::Reserved);
    if (!cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slots_[token.slot].ownership}
             .compare_exchange_strong(expected, ownership(token.slot_generation, Phase::Recycling),
                                      cuda::memory_order_acq_rel, cuda::memory_order_acquire)) {
      return committed(expected, token.slot_generation) ? AbandonStatus::AlreadyCommitted : AbandonStatus::StaleToken;
    }
    recycle(token.slot, token.slot_generation);
    return AbandonStatus::Abandoned;
  }

  /// Acquire one immutable committed generation, skipping all pending holes.
  /// A successful result holds Copying ownership until explicit release.
  XPOOL_DEVICE_FN ReadResult try_acquire() const {
    if (slots_.empty()) {
      return {};
    }
    const auto cursor = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.consume_cursor}.fetch_add(
        std::uint64_t{1}, cuda::memory_order_relaxed);
    auto index = static_cast<std::size_t>(cursor % slots_.size());
    for (auto inspected = std::size_t{0}; inspected < slots_.size(); ++inspected) {
      const auto result = try_acquire_at(index);
      if (result.acquired) {
        return result;
      }
      index = index + 1 == slots_.size() ? 0 : index + 1;
    }
    return {};
  }

  /// Inspect one candidate in a process driver's bounded scan. A single
  /// atomic claim grants immutable Copying ownership; invalid/occupied slots
  /// return empty. This avoids repeated full scans while packing a batch.
  XPOOL_DEVICE_FN ReadResult try_acquire_at(std::size_t index) const {
    if (index >= slots_.size()) {
      return {};
    }
    auto &slot = slots_[index];
    auto word = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.ownership};
    auto expected = word.load(cuda::memory_order_acquire);
    const auto generation = expected >> 4;
    if (phase_of(expected) == Phase::Committed &&
        word.compare_exchange_strong(expected, ownership(generation, Phase::Copying), cuda::memory_order_acq_rel,
                                     cuda::memory_order_acquire)) {
      const auto sequence = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slot.event_sequence}.load(
          cuda::memory_order_relaxed);
      return {.acquired = true,
              .token = {.slot = index, .slot_generation = generation, .event_sequence = sequence},
              .event = &events_[index]};
    }
    return {};
  }

  /// Retire read ownership only after every payload read and asynchronous copy
  /// has completed. Submission or timeout is not completion. A management
  /// kernel may call this after the Collector observes its D2H completion event.
  XPOOL_DEVICE_FN ReleaseStatus release(ReadToken token) const {
    const auto write = WriteToken{token.slot, token.slot_generation, token.event_sequence};
    if (!matches(write)) {
      return ReleaseStatus::StaleToken;
    }
    auto expected = ownership(token.slot_generation, Phase::Copying);
    if (!cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slots_[token.slot].ownership}
             .compare_exchange_strong(expected, ownership(token.slot_generation, Phase::Recycling),
                                      cuda::memory_order_acq_rel, cuda::memory_order_acquire)) {
      return (expected >> 4) == token.slot_generation ? ReleaseStatus::NotAcquired : ReleaseStatus::StaleToken;
    }
    recycle(token.slot, token.slot_generation);
    return ReleaseStatus::Released;
  }

  /// Close new admission on this Device. Existing reservations and copy owners
  /// remain valid. A reservation already in progress may finish its admission.
  XPOOL_DEVICE_FN void stop() const {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.stopped}.fetch_or(stop_requested,
                                                                                        cuda::memory_order_release);
  }

private:
  enum class Phase : std::uint64_t {
    Free = 0,
    Initializing = 1,
    Reserved = 3,
    Writing = 5,
    Committed = 7,
    Copying = 9,
    Recycling = 11,
    Retired = 13,
  };
  static constexpr std::uint64_t stop_requested = 1;
  static constexpr std::uint64_t generation_exhausted = 2;

  XPOOL_DEVICE_FN static std::uint64_t ownership(std::uint64_t generation, Phase phase) {
    return (generation << 4) | static_cast<std::uint64_t>(phase);
  }
  XPOOL_DEVICE_FN static Phase phase_of(std::uint64_t value) { return static_cast<Phase>(value & std::uint64_t{15}); }
  XPOOL_DEVICE_FN static bool committed(std::uint64_t value, std::uint64_t generation) {
    const auto phase = phase_of(value);
    return (value >> 4) == generation &&
           (phase == Phase::Writing || phase == Phase::Committed || phase == Phase::Copying);
  }
  XPOOL_DEVICE_FN bool matches(WriteToken token) const {
    return token.slot < slots_.size() && token.slot_generation != 0 && token.slot_generation <= kSlotGenerationLimit &&
           token.event_sequence != 0 &&
           cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slots_[token.slot].event_sequence}.load(
               cuda::memory_order_relaxed) == token.event_sequence;
  }
  XPOOL_DEVICE_FN void recycle(std::size_t slot, std::uint64_t generation) const {
    auto word = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{slots_[slot].ownership};
    if (generation == kSlotGenerationLimit) {
      word.store(ownership(generation, Phase::Retired), cuda::memory_order_release);
      cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state_.stopped}.fetch_or(generation_exhausted,
                                                                                          cuda::memory_order_release);
    } else {
      word.store(ownership(generation + 1, Phase::Free), cuda::memory_order_release);
    }
  }

  cuda::std::span<Event> events_;
  cuda::std::span<SlotControl> slots_;
  SlotPoolState &state_;
};

} // namespace xpool::devkit::timeline
