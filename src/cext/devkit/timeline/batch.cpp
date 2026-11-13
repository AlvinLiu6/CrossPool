#include <xpool/devkit/timeline/batch.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <limits>
#include <stdexcept>

namespace xpool::devkit::timeline {

namespace {

enum class Phase : std::uint64_t {
  Free = 0,
  Filling = 1,
  Publishing = 3,
  Ready = 5,
  Reading = 7,
  Accounting = 9,
  Finalized = 11,
  Recycling = 13,
  Retired = 15,
};

constexpr auto stop_requested = std::uint64_t{1};
constexpr auto generation_exhausted = std::uint64_t{2};

constexpr std::uint64_t ownership(std::uint64_t generation, Phase phase) {
  return (generation << 4) | static_cast<std::uint64_t>(phase);
}

constexpr Phase phase_of(std::uint64_t value) { return static_cast<Phase>(value & std::uint64_t{15}); }

bool published(std::uint64_t value, std::uint64_t generation) {
  const auto phase = phase_of(value);
  return (value >> 4) == generation &&
         (phase == Phase::Publishing || phase == Phase::Ready || phase == Phase::Reading ||
          phase == Phase::Accounting || phase == Phase::Finalized);
}

std::size_t checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first) {
    throw std::overflow_error("xpool batch layout addition overflows");
  }
  return first + second;
}

std::size_t checked_product(std::size_t first, std::size_t second) {
  if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second) {
    throw std::overflow_error("xpool batch layout multiplication overflows");
  }
  return first * second;
}

std::size_t aligned(std::size_t value, std::size_t alignment) {
  return checked_add(value, (alignment - value % alignment) % alignment);
}

bool overlaps(const void *first, std::size_t first_bytes, const void *second, std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}

void validate_dimensions(std::size_t batches, std::size_t events) {
  if (batches == 0 || events == 0 || events > Chunk::maximum_events) {
    throw std::invalid_argument("xpool batch pool requires positive batches and 1 through 4095 events per batch");
  }
}

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
static_assert(alignof(BatchControl) >= std::atomic_ref<std::uint64_t>::required_alignment);
static_assert(alignof(BatchPoolState) >= std::atomic_ref<std::uint64_t>::required_alignment);

} // namespace

Chunk BatchView::chunk(std::uint64_t chunk_sequence) const {
  if (chunk_sequence == 0) {
    throw std::invalid_argument("xpool batch chunk sequence must be positive");
  }
  return {.producer = source.producer,
          .chunk_sequence = chunk_sequence,
          .metadata_revision = info.metadata_revision,
          .host_window_begin_ns = info.host_window_begin_ns,
          .host_window_end_ns = info.host_window_end_ns,
          .run_id = source.run_id,
          .metadata_sha256_prefix = info.metadata_sha256_prefix,
          .events = events};
}

HostBatchPool::HostBatchPool(BatchSource source, std::size_t events_per_batch, std::span<Event> events,
                             std::span<BatchControl> batches, BatchPoolState &state)
    : source_(source), events_per_batch_(events_per_batch), events_(events), batches_(batches), state_(state) {
  validate_dimensions(batches.size(), events_per_batch);
  if (source.producer == 0 || source.clock_domain_id == 0 ||
      events.size() != checked_product(batches.size(), events_per_batch)) {
    throw std::invalid_argument("xpool batch pool requires a registered source and exact payload geometry");
  }
  const auto event_bytes = checked_product(events.size(), sizeof(Event));
  const auto control_bytes = checked_product(batches.size(), sizeof(BatchControl));
  if (overlaps(events.data(), event_bytes, batches.data(), control_bytes) ||
      overlaps(events.data(), event_bytes, &state, sizeof(state)) ||
      overlaps(batches.data(), control_bytes, &state, sizeof(state))) {
    throw std::invalid_argument("xpool batch pool storage regions overlap");
  }
  if (state.write_cursor != 0 || state.read_cursor != 0 || state.stopped != 0 || state.copied.value() != 0 ||
      state.exported.value() != 0 || state.post_commit_lost.value() != 0 || state.copied.exhausted() ||
      state.exported.exhausted() || state.post_commit_lost.exhausted() ||
      std::ranges::any_of(batches, [](const auto &batch) {
        return phase_of(batch.ownership) != Phase::Free || (batch.ownership >> 4) == 0 || batch.event_count != 0;
      })) {
    throw std::invalid_argument("xpool batch pool requires free controls and zero lane state");
  }
}

bool HostBatchPool::valid_token(std::uint64_t batch, std::uint64_t generation) const {
  return batch < batches_.size() && generation != 0 && generation <= BatchControl::generation_limit;
}

BatchReservation HostBatchPool::try_reserve() {
  const auto flags = std::atomic_ref{state_.stopped}.load(std::memory_order_acquire);
  if (flags != 0) {
    return {.status = (flags & generation_exhausted) != 0 ? BatchReserveStatus::IdentityExhausted
                                                          : BatchReserveStatus::Stopped};
  }
  auto index = state_.write_cursor;
  for (auto inspected = std::size_t{0}; inspected < batches_.size(); ++inspected) {
    auto &batch = batches_[index];
    // Every occupied phase is odd. One RMW claims Free and leaves occupied
    // phases unchanged, including ownership held by a slow Writer.
    const auto previous = std::atomic_ref{batch.ownership}.fetch_or(std::uint64_t{1}, std::memory_order_acquire);
    if (phase_of(previous) == Phase::Free) {
      batch.event_count = 0;
      batch.info = {};
      state_.write_cursor = index + 1 == batches_.size() ? 0 : index + 1;
      return {.status = BatchReserveStatus::Reserved,
              .token = {.batch = index, .generation = previous >> 4},
              .events = events_.subspan(index * events_per_batch_, events_per_batch_)};
    }
    index = index + 1 == batches_.size() ? 0 : index + 1;
  }
  return {.status = BatchReserveStatus::Full};
}

BatchPublishStatus HostBatchPool::publish(BatchWriteToken token, std::size_t event_count, BatchInfo info) {
  if (!valid_token(token.batch, token.generation)) {
    return BatchPublishStatus::StaleToken;
  }
  auto &batch = batches_[token.batch];
  auto expected = ownership(token.generation, Phase::Filling);
  if (!std::atomic_ref{batch.ownership}.compare_exchange_strong(expected,
                                                                ownership(token.generation, Phase::Publishing),
                                                                std::memory_order_acq_rel, std::memory_order_acquire)) {
    return published(expected, token.generation) ? BatchPublishStatus::AlreadyPublished
                                                 : BatchPublishStatus::StaleToken;
  }
  try {
    if (event_count == 0 || event_count > events_per_batch_ || info.metadata_revision == 0 ||
        info.host_window_end_ns < info.host_window_begin_ns) {
      throw std::invalid_argument("xpool batch count or metadata/window is invalid");
    }
    auto previous = std::uint64_t{0};
    for (const auto &event : events_.subspan(token.batch * events_per_batch_, event_count)) {
      if (event.clock_domain_id != source_.clock_domain_id || event.event_sequence <= previous) {
        throw std::invalid_argument("xpool batch requires its source clock and positive increasing event sequences");
      }
      previous = event.event_sequence;
    }
  } catch (...) {
    // Validation is the sole throwing operation while this generation is
    // exclusively Publishing. Restore its fill authority before propagating.
    std::atomic_ref{batch.ownership}.store(ownership(token.generation, Phase::Filling), std::memory_order_release);
    throw;
  }
  batch.event_count = event_count;
  batch.info = info;
  state_.copied.add(static_cast<std::uint64_t>(event_count));
  std::atomic_ref{batch.ownership}.store(ownership(token.generation, Phase::Ready), std::memory_order_release);
  return BatchPublishStatus::Published;
}

void HostBatchPool::recycle(std::size_t batch, std::uint64_t generation) {
  batches_[batch].event_count = 0;
  if (generation == BatchControl::generation_limit) {
    std::atomic_ref{batches_[batch].ownership}.store(ownership(generation, Phase::Retired), std::memory_order_release);
    std::atomic_ref{state_.stopped}.fetch_or(generation_exhausted, std::memory_order_release);
  } else {
    std::atomic_ref{batches_[batch].ownership}.store(ownership(generation + 1, Phase::Free), std::memory_order_release);
  }
}

BatchCancelStatus HostBatchPool::cancel(BatchWriteToken token) {
  if (!valid_token(token.batch, token.generation)) {
    return BatchCancelStatus::StaleToken;
  }
  auto expected = ownership(token.generation, Phase::Filling);
  if (!std::atomic_ref{batches_[token.batch].ownership}.compare_exchange_strong(
          expected, ownership(token.generation, Phase::Recycling), std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return published(expected, token.generation) ? BatchCancelStatus::AlreadyPublished : BatchCancelStatus::StaleToken;
  }
  recycle(token.batch, token.generation);
  return BatchCancelStatus::Canceled;
}

BatchAcquisition HostBatchPool::try_acquire() {
  auto index = state_.read_cursor;
  for (auto inspected = std::size_t{0}; inspected < batches_.size(); ++inspected) {
    auto &batch = batches_[index];
    auto expected = std::atomic_ref{batch.ownership}.load(std::memory_order_acquire);
    const auto generation = expected >> 4;
    if (phase_of(expected) == Phase::Ready &&
        std::atomic_ref{batch.ownership}.compare_exchange_strong(
            expected, ownership(generation, Phase::Reading), std::memory_order_acq_rel, std::memory_order_acquire)) {
      state_.read_cursor = index + 1 == batches_.size() ? 0 : index + 1;
      return {.acquired = true,
              .token = {.batch = index, .generation = generation},
              .view = {.source = source_,
                       .info = batch.info,
                       .events = events_.subspan(index * events_per_batch_, batch.event_count)}};
    }
    index = index + 1 == batches_.size() ? 0 : index + 1;
  }
  return {};
}

BatchReportStatus HostBatchPool::report(BatchReadToken token, BatchOutcome outcome) {
  if (!valid_token(token.batch, token.generation)) {
    return BatchReportStatus::StaleToken;
  }
  auto &batch = batches_[token.batch];
  auto expected = ownership(token.generation, Phase::Reading);
  if (!std::atomic_ref{batch.ownership}.compare_exchange_strong(expected,
                                                                ownership(token.generation, Phase::Accounting),
                                                                std::memory_order_acq_rel, std::memory_order_acquire)) {
    if ((expected >> 4) != token.generation || phase_of(expected) == Phase::Retired) {
      return BatchReportStatus::StaleToken;
    }
    return phase_of(expected) == Phase::Finalized || phase_of(expected) == Phase::Accounting
               ? BatchReportStatus::AlreadyRecorded
               : BatchReportStatus::NotAcquired;
  }
  const auto count = static_cast<std::uint64_t>(batch.event_count);
  if (outcome.exported > count || outcome.post_commit_lost != count - outcome.exported) {
    std::atomic_ref{batch.ownership}.store(ownership(token.generation, Phase::Reading), std::memory_order_release);
    throw std::invalid_argument("xpool batch outcome must classify every event exactly once");
  }
  state_.exported.add(outcome.exported);
  state_.post_commit_lost.add(outcome.post_commit_lost);
  std::atomic_ref{batch.ownership}.store(ownership(token.generation, Phase::Finalized), std::memory_order_release);
  return BatchReportStatus::Recorded;
}

BatchReleaseStatus HostBatchPool::release(BatchReadToken token) {
  if (!valid_token(token.batch, token.generation)) {
    return BatchReleaseStatus::StaleToken;
  }
  auto expected = ownership(token.generation, Phase::Finalized);
  if (!std::atomic_ref{batches_[token.batch].ownership}.compare_exchange_strong(
          expected, ownership(token.generation, Phase::Recycling), std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    if ((expected >> 4) != token.generation || phase_of(expected) == Phase::Retired) {
      return BatchReleaseStatus::StaleToken;
    }
    return phase_of(expected) == Phase::Reading || phase_of(expected) == Phase::Accounting
               ? BatchReleaseStatus::OutcomeMissing
               : BatchReleaseStatus::NotAcquired;
  }
  recycle(token.batch, token.generation);
  return BatchReleaseStatus::Released;
}

void HostBatchPool::stop() { std::atomic_ref{state_.stopped}.fetch_or(stop_requested, std::memory_order_release); }

BatchCounts HostBatchPool::quiescent_counts() const {
  const auto flags = std::atomic_ref{state_.stopped}.load(std::memory_order_acquire);
  auto result = BatchCounts{.copied = state_.copied.value(),
                            .exported = state_.exported.value(),
                            .post_commit_lost = state_.post_commit_lost.value(),
                            .counter_exhausted = state_.copied.exhausted() || state_.exported.exhausted() ||
                                                 state_.post_commit_lost.exhausted(),
                            .stopped = flags != 0,
                            .identity_exhausted = (flags & generation_exhausted) != 0};
  for (const auto &batch : batches_) {
    switch (phase_of(batch.ownership)) {
    case Phase::Free:
      ++result.free;
      break;
    case Phase::Filling:
      ++result.filling;
      break;
    case Phase::Ready:
      ++result.ready;
      break;
    case Phase::Reading:
      ++result.reading;
      break;
    case Phase::Finalized:
      ++result.finalized;
      break;
    case Phase::Retired:
      ++result.retired;
      break;
    default:
      throw std::logic_error("xpool batch counts require quiescent lane calls");
    }
  }
  return result;
}

BatchPoolLayout HostBatchPool::layout(std::size_t batch_count, std::size_t events_per_batch,
                                      std::size_t allocation_alignment) {
  validate_dimensions(batch_count, events_per_batch);
  const auto minimum_alignment = std::max({alignof(Event), alignof(BatchControl), alignof(BatchPoolState)});
  if (!std::has_single_bit(allocation_alignment) || allocation_alignment < minimum_alignment) {
    throw std::invalid_argument("xpool batch pool allocation alignment is invalid");
  }
  const auto event_bytes = checked_product(checked_product(batch_count, events_per_batch), sizeof(Event));
  const auto control_offset = aligned(event_bytes, alignof(BatchControl));
  const auto control_bytes = checked_product(batch_count, sizeof(BatchControl));
  const auto state_offset = aligned(checked_add(control_offset, control_bytes), alignof(BatchPoolState));
  return {.event_bytes = event_bytes,
          .control_offset = control_offset,
          .control_bytes = control_bytes,
          .state_offset = state_offset,
          .state_bytes = sizeof(BatchPoolState),
          .total_bytes = aligned(checked_add(state_offset, sizeof(BatchPoolState)), allocation_alignment)};
}

} // namespace xpool::devkit::timeline
