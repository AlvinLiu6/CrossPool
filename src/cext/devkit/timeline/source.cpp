#include <xpool/devkit/timeline/source.hpp>

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>

namespace xpool::devkit::timeline {

namespace {

constexpr auto stopped = std::uint64_t{1} << 62;
constexpr auto exhausted = std::uint64_t{1} << 63;
constexpr auto active_mask = stopped - 1;
static_assert(kAttemptConcurrencyLimit < active_mask);

bool overlaps(const void *first, std::size_t first_bytes, const void *second, std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("xpool source layout overflows");
  }
  return left + right;
}

std::size_t aligned(std::size_t value, std::size_t alignment) {
  return checked_add(value, (alignment - value % alignment) % alignment);
}

bool conserved(const SourceCounts &counts) {
  auto remaining = counts.attempted;
  for (const auto value : {counts.committed, counts.dropped_full, counts.abandoned, counts.pending}) {
    if (value > remaining) {
      return false;
    }
    remaining -= value;
  }
  return remaining == 0;
}

SourceLayout source_layout(std::size_t capacity, std::size_t state_bytes, std::size_t state_alignment,
                           std::size_t allocation_alignment) {
  const auto minimum = std::max({alignof(Event), alignof(SlotControl), alignof(SlotPoolState), state_alignment});
  if (allocation_alignment < minimum) {
    throw std::invalid_argument("xpool source allocation alignment is too small");
  }
  const auto slots = HostSlotPool::layout(capacity, minimum);
  HostSlotPool::layout(capacity, allocation_alignment);
  const auto offset = aligned(slots.total_bytes, state_alignment);
  return {.slots = slots,
          .source_state_offset = offset,
          .source_state_bytes = state_bytes,
          .total_bytes = aligned(checked_add(offset, state_bytes), allocation_alignment)};
}

} // namespace

HostEventSource::HostEventSource(BatchSource source, std::span<Event> events, std::span<SlotControl> slots,
                                 SlotPoolState &pool_state, SourceState &state)
    : source_(source), pool_(events, slots, pool_state), state_(state) {
  if (source.producer == 0 || source.clock_domain_id == 0 ||
      overlaps(&state, sizeof(state), events.data(), events.size_bytes()) ||
      overlaps(&state, sizeof(state), slots.data(), slots.size_bytes()) ||
      overlaps(&state, sizeof(state), &pool_state, sizeof(pool_state)) ||
      (state.admission != 0 && state.admission != stopped && state.admission != (stopped | exhausted))) {
    throw std::invalid_argument("xpool source requires registered identity and initialized disjoint storage");
  }
  auto counts = SourceCounts{};
  try {
    counts = quiescent_counts();
  } catch (const std::logic_error &) {
    throw std::invalid_argument("xpool source history violates attempt conservation");
  }
  if (!conserved(counts)) {
    throw std::invalid_argument("xpool source history violates attempt conservation");
  }
  if (counts.last_attempted_sequence == std::numeric_limits<std::uint64_t>::max()) {
    state_.admission |= stopped | exhausted;
  }
}

ReserveResult HostEventSource::try_reserve() {
  auto gate = std::atomic_ref{state_.admission};
  const auto previous = gate.fetch_add(std::uint64_t{1}, std::memory_order_acquire);
  if ((previous & (stopped | exhausted)) != 0) {
    gate.fetch_sub(std::uint64_t{1}, std::memory_order_release);
    return {.status = (previous & exhausted) != 0 ? ReserveStatus::IdentityExhausted : ReserveStatus::Stopped};
  }
  if ((previous & active_mask) >= kAttemptConcurrencyLimit) {
    gate.fetch_sub(std::uint64_t{1}, std::memory_order_release);
    throw std::overflow_error("xpool source reservation concurrency exceeds its supported bound");
  }
  const auto sequence = state_.attempts.try_next();
  if (sequence == 0) {
    gate.fetch_or(stopped | exhausted, std::memory_order_relaxed);
    gate.fetch_sub(std::uint64_t{1}, std::memory_order_release);
    return {.status = ReserveStatus::IdentityExhausted};
  }
  const auto result = pool_.reserve_admitted(sequence);
  if (result.status == ReserveStatus::Full) {
    state_.dropped_full.add(1);
  }
  if (sequence == std::numeric_limits<std::uint64_t>::max()) {
    gate.fetch_or(stopped | exhausted, std::memory_order_relaxed);
  }
  gate.fetch_sub(std::uint64_t{1}, std::memory_order_release);
  return result;
}

CommitStatus HostEventSource::commit(WriteToken token, const Event &event) {
  if (event.clock_domain_id != source_.clock_domain_id) {
    throw std::invalid_argument("xpool event clock differs from its registered source");
  }
  const auto result = pool_.commit(token, event);
  if (result == CommitStatus::Committed) {
    state_.committed.add(1);
  } else if (result == CommitStatus::AlreadyCommitted) {
    state_.duplicate_commit.add(1);
  } else {
    state_.stale_token.add(1);
  }
  return result;
}

AbandonStatus HostEventSource::abandon(WriteToken token) {
  const auto result = pool_.abandon(token);
  if (result == AbandonStatus::Abandoned) {
    state_.abandoned.add(1);
    if (token.slot_generation == kSlotGenerationLimit) {
      std::atomic_ref{state_.admission}.fetch_or(stopped | exhausted, std::memory_order_release);
    }
  } else {
    state_.stale_token.add(1);
  }
  return result;
}

ReadResult HostEventSource::try_acquire() { return pool_.try_acquire(); }

ReleaseStatus HostEventSource::release(ReadToken token) {
  const auto result = pool_.release(token);
  if (result != ReleaseStatus::Released) {
    state_.stale_token.add(1);
  } else if (token.slot_generation == kSlotGenerationLimit) {
    std::atomic_ref{state_.admission}.fetch_or(stopped | exhausted, std::memory_order_release);
  }
  return result;
}

void HostEventSource::stop() { std::atomic_ref{state_.admission}.fetch_or(stopped, std::memory_order_release); }

SourceProgress HostEventSource::progress() const {
  const auto gate = std::atomic_ref{state_.admission}.load(std::memory_order_acquire);
  const auto committed = state_.committed.lower_bound();
  const auto full = state_.dropped_full.lower_bound();
  const auto abandoned = state_.abandoned.lower_bound();
  const auto stale = state_.stale_token.lower_bound();
  const auto duplicate = state_.duplicate_commit.lower_bound();
  const auto attempted = state_.attempts.lower_bound_value();
  return {.counts = {.attempted = attempted,
                     .last_attempted_sequence = attempted,
                     .committed = committed.value,
                     .dropped_full = full.value,
                     .abandoned = abandoned.value,
                     .stale_token = stale.value,
                     .duplicate_commit = duplicate.value,
                     .counter_exhausted = committed.exhausted || full.exhausted || abandoned.exhausted ||
                                          stale.exhausted || duplicate.exhausted,
                     .exact_totals = false,
                     .stopped = (gate & stopped) != 0,
                     .identity_exhausted = (gate & exhausted) != 0},
          .pending_known = false};
}

SourceCounts HostEventSource::quiescent_counts() const {
  const auto gate = std::atomic_ref{state_.admission}.load(std::memory_order_acquire);
  if ((gate & active_mask) != 0) {
    throw std::logic_error("xpool source counts require quiescent API calls");
  }
  const auto slots = pool_.quiescent_counts();
  const auto committed = state_.committed.quiescent();
  const auto full = state_.dropped_full.quiescent();
  const auto abandoned = state_.abandoned.quiescent();
  const auto stale = state_.stale_token.quiescent();
  const auto duplicate = state_.duplicate_commit.quiescent();
  const auto saturated =
      committed.exhausted || full.exhausted || abandoned.exhausted || stale.exhausted || duplicate.exhausted;
  const auto attempted = state_.attempts.quiescent_value();
  const auto result = SourceCounts{.attempted = attempted,
                                   .last_attempted_sequence = attempted,
                                   .committed = committed.value,
                                   .dropped_full = full.value,
                                   .abandoned = abandoned.value,
                                   .stale_token = stale.value,
                                   .duplicate_commit = duplicate.value,
                                   .pending = static_cast<std::uint64_t>(slots.reserved),
                                   .counter_exhausted = saturated,
                                   .exact_totals = !saturated,
                                   .stopped = (gate & stopped) != 0 || slots.stopped,
                                   .identity_exhausted = (gate & exhausted) != 0 || slots.identity_exhausted};
  if (!saturated && !conserved(result)) {
    throw std::logic_error("xpool source totals require quiescent outcome calls");
  }
  return result;
}

SourceLayout HostEventSource::layout(std::size_t capacity, std::size_t allocation_alignment) {
  return source_layout(capacity, sizeof(SourceState), alignof(SourceState), allocation_alignment);
}

SourceLayout DeviceSourceState::layout(std::size_t capacity, std::size_t allocation_alignment) {
  return source_layout(capacity, sizeof(DeviceSourceState), alignof(DeviceSourceState), allocation_alignment);
}

} // namespace xpool::devkit::timeline
