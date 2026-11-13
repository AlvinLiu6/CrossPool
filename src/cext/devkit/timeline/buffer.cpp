#include <xpool/devkit/timeline/buffer.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <stdexcept>

namespace xpool::devkit::timeline {

namespace {

// Every occupied phase is odd: one fetch-or claims Free without modifying a
// contended slot. Initialization has its own phase so tokens cannot claim a
// slot before its sequence has been installed.
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

constexpr auto phase_mask = std::uint64_t{15};
constexpr auto stop_requested = std::uint64_t{1};
constexpr auto generation_exhausted = std::uint64_t{2};

constexpr std::uint64_t ownership(std::uint64_t generation, Phase phase) {
  return (generation << 4) | static_cast<std::uint64_t>(phase);
}

constexpr Phase phase_of(std::uint64_t value) { return static_cast<Phase>(value & phase_mask); }

bool committing(std::uint64_t value, std::uint64_t generation) {
  return (value >> 4) == generation && (phase_of(value) == Phase::Writing || phase_of(value) == Phase::Committed ||
                                        phase_of(value) == Phase::Copying);
}

std::size_t checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    throw std::overflow_error("xpool slot pool layout addition overflows");
  }
  return first + second;
}

std::size_t checked_product(std::size_t count, std::size_t size) {
  if (count > std::numeric_limits<std::size_t>::max() / size) {
    throw std::overflow_error("xpool slot pool layout multiplication overflows");
  }
  return count * size;
}

std::size_t aligned(std::size_t value, std::size_t alignment) {
  const auto padding = (alignment - value % alignment) % alignment;
  return checked_add(value, padding);
}

bool overlaps(const void *first, std::size_t first_bytes, const void *second, std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
static_assert(alignof(SlotControl) >= std::atomic_ref<std::uint64_t>::required_alignment);
static_assert(alignof(SlotPoolState) >= std::atomic_ref<std::uint64_t>::required_alignment);

} // namespace

HostSlotPool::HostSlotPool(std::span<Event> events, std::span<SlotControl> slots, SlotPoolState &state)
    : events_(events), slots_(slots), state_(state) {
  if (events.empty() || events.size() != slots.size()) {
    throw std::invalid_argument("xpool slot pool requires equal nonempty event and control spans");
  }
  const auto event_bytes = checked_product(events.size(), sizeof(Event));
  const auto control_bytes = checked_product(slots.size(), sizeof(SlotControl));
  if (overlaps(events.data(), event_bytes, slots.data(), control_bytes) ||
      overlaps(events.data(), event_bytes, &state, sizeof(state)) ||
      overlaps(slots.data(), control_bytes, &state, sizeof(state))) {
    throw std::invalid_argument("xpool slot pool storage regions overlap");
  }
  if (state.consume_cursor != 0 || state.stopped != 0 || std::ranges::any_of(slots, [](const auto &slot) {
        return phase_of(slot.ownership) != Phase::Free || (slot.ownership >> 4) == 0 || slot.event_sequence != 0;
      })) {
    throw std::invalid_argument("xpool slot pool requires initialized free controls and zero scheduling state");
  }
}

ReserveResult HostSlotPool::try_reserve(std::uint64_t event_sequence) {
  if (event_sequence == 0) {
    throw std::invalid_argument("xpool slot pool event sequence must be positive");
  }
  const auto stop_flags = std::atomic_ref{state_.stopped}.load(std::memory_order_acquire);
  if (stop_flags != 0) {
    return {.status =
                (stop_flags & generation_exhausted) != 0 ? ReserveStatus::IdentityExhausted : ReserveStatus::Stopped};
  }
  return reserve_admitted(event_sequence);
}

ReserveResult HostSlotPool::reserve_admitted(std::uint64_t event_sequence) {
  auto index = static_cast<std::size_t>((event_sequence - 1) % slots_.size());
  for (auto inspected = std::size_t{0}; inspected < slots_.size(); ++inspected) {
    auto &slot = slots_[index];
    const auto previous = std::atomic_ref{slot.ownership}.fetch_or(std::uint64_t{1}, std::memory_order_acquire);
    if (phase_of(previous) == Phase::Free) {
      const auto generation = previous >> 4;
      std::atomic_ref{slot.event_sequence}.store(event_sequence, std::memory_order_relaxed);
      std::atomic_ref{slot.ownership}.store(ownership(generation, Phase::Reserved), std::memory_order_release);
      return {.status = ReserveStatus::Reserved,
              .token = {.slot = index, .slot_generation = generation, .event_sequence = event_sequence}};
    }
    index = index + 1 == slots_.size() ? 0 : index + 1;
  }
  return {.status = ReserveStatus::Full};
}

bool HostSlotPool::matches(WriteToken token) const {
  return token.slot < slots_.size() && token.slot_generation != 0 && token.slot_generation <= kSlotGenerationLimit &&
         token.event_sequence != 0 &&
         std::atomic_ref{slots_[token.slot].event_sequence}.load(std::memory_order_relaxed) == token.event_sequence;
}

CommitStatus HostSlotPool::commit(WriteToken token, const Event &event) {
  if (!matches(token)) {
    return CommitStatus::StaleToken;
  }
  auto &slot = slots_[token.slot];
  auto expected = ownership(token.slot_generation, Phase::Reserved);
  if (!std::atomic_ref{slot.ownership}.compare_exchange_strong(expected,
                                                               ownership(token.slot_generation, Phase::Writing),
                                                               std::memory_order_acq_rel, std::memory_order_acquire)) {
    return committing(expected, token.slot_generation) ? CommitStatus::AlreadyCommitted : CommitStatus::StaleToken;
  }
  events_[token.slot] = event;
  events_[token.slot].event_sequence = token.event_sequence;
  std::atomic_ref{slot.ownership}.store(ownership(token.slot_generation, Phase::Committed), std::memory_order_release);
  return CommitStatus::Committed;
}

void HostSlotPool::recycle(std::size_t slot, std::uint64_t generation) {
  if (generation == kSlotGenerationLimit) {
    std::atomic_ref{slots_[slot].ownership}.store(ownership(generation, Phase::Retired), std::memory_order_release);
    std::atomic_ref{state_.stopped}.fetch_or(generation_exhausted, std::memory_order_release);
    return;
  }
  // This release pairs with the next reservation's acquire. A previous copy
  // owner's complete payload reads precede any writes by the next generation.
  std::atomic_ref{slots_[slot].ownership}.store(ownership(generation + 1, Phase::Free), std::memory_order_release);
}

AbandonStatus HostSlotPool::abandon(WriteToken token) {
  if (!matches(token)) {
    return AbandonStatus::StaleToken;
  }
  auto expected = ownership(token.slot_generation, Phase::Reserved);
  if (!std::atomic_ref{slots_[token.slot].ownership}.compare_exchange_strong(
          expected, ownership(token.slot_generation, Phase::Recycling), std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return committing(expected, token.slot_generation) ? AbandonStatus::AlreadyCommitted : AbandonStatus::StaleToken;
  }
  recycle(token.slot, token.slot_generation);
  return AbandonStatus::Abandoned;
}

ReadResult HostSlotPool::try_acquire() {
  auto index = static_cast<std::size_t>(std::atomic_ref{state_.consume_cursor}.fetch_add(1, std::memory_order_relaxed) %
                                        slots_.size());
  for (auto inspected = std::size_t{0}; inspected < slots_.size(); ++inspected) {
    auto &slot = slots_[index];
    auto expected = std::atomic_ref{slot.ownership}.load(std::memory_order_acquire);
    const auto generation = expected >> 4;
    if (phase_of(expected) == Phase::Committed &&
        std::atomic_ref{slot.ownership}.compare_exchange_strong(expected, ownership(generation, Phase::Copying),
                                                                std::memory_order_acq_rel, std::memory_order_acquire)) {
      return {.acquired = true,
              .token = {.slot = index,
                        .slot_generation = generation,
                        .event_sequence = std::atomic_ref{slot.event_sequence}.load(std::memory_order_relaxed)},
              .event = &events_[index]};
    }
    index = index + 1 == slots_.size() ? 0 : index + 1;
  }
  return {};
}

ReleaseStatus HostSlotPool::release(ReadToken token) {
  if (!matches(
          {.slot = token.slot, .slot_generation = token.slot_generation, .event_sequence = token.event_sequence})) {
    return ReleaseStatus::StaleToken;
  }
  auto expected = ownership(token.slot_generation, Phase::Copying);
  if (!std::atomic_ref{slots_[token.slot].ownership}.compare_exchange_strong(
          expected, ownership(token.slot_generation, Phase::Recycling), std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return (expected >> 4) == token.slot_generation ? ReleaseStatus::NotAcquired : ReleaseStatus::StaleToken;
  }
  recycle(token.slot, token.slot_generation);
  return ReleaseStatus::Released;
}

void HostSlotPool::stop() { std::atomic_ref{state_.stopped}.fetch_or(stop_requested, std::memory_order_release); }

SlotCounts HostSlotPool::quiescent_counts() const {
  auto result = SlotCounts{};
  const auto stop_flags = std::atomic_ref{state_.stopped}.load(std::memory_order_acquire);
  result.stopped = stop_flags != 0;
  result.identity_exhausted = (stop_flags & generation_exhausted) != 0;
  for (const auto &slot : slots_) {
    switch (phase_of(slot.ownership)) {
    case Phase::Free:
      ++result.free;
      break;
    case Phase::Reserved:
      ++result.reserved;
      break;
    case Phase::Committed:
      ++result.committed;
      break;
    case Phase::Copying:
      ++result.copying;
      break;
    case Phase::Retired:
      ++result.retired;
      break;
    default:
      throw std::logic_error("xpool slot pool occupancy requires quiescent calls");
    }
  }
  return result;
}

SlotPoolLayout HostSlotPool::layout(std::size_t capacity, std::size_t allocation_alignment) {
  const auto minimum_alignment = std::max({alignof(Event), alignof(SlotControl), alignof(SlotPoolState)});
  if (capacity == 0 || !std::has_single_bit(allocation_alignment) || allocation_alignment < minimum_alignment) {
    throw std::invalid_argument("xpool slot pool requires positive capacity and valid allocation alignment");
  }
  const auto event_bytes = checked_product(capacity, sizeof(Event));
  const auto control_offset = aligned(event_bytes, alignof(SlotControl));
  const auto control_bytes = checked_product(capacity, sizeof(SlotControl));
  const auto state_offset = aligned(checked_add(control_offset, control_bytes), alignof(SlotPoolState));
  return {.event_bytes = event_bytes,
          .control_offset = control_offset,
          .control_bytes = control_bytes,
          .state_offset = state_offset,
          .state_bytes = sizeof(SlotPoolState),
          .total_bytes = aligned(checked_add(state_offset, sizeof(SlotPoolState)), allocation_alignment)};
}

} // namespace xpool::devkit::timeline
