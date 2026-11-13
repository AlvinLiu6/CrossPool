#include <xpool/devkit/timeline/collector.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>

namespace xpool::devkit::timeline {

namespace {

constexpr auto source_stopped = std::uint32_t{1};
constexpr auto source_exhausted = std::uint32_t{2};
constexpr auto source_saturated = std::uint32_t{4};
constexpr auto external_count = static_cast<std::size_t>(ExternalCounter::Count);
constexpr std::uint64_t QualityCounters::*external_fields[external_count] = {
    &QualityCounters::snapshot_operations_attempted,
    &QualityCounters::snapshot_operations_dropped,
    &QualityCounters::attachment_attempted,
    &QualityCounters::attachment_capacity_exhausted,
    &QualityCounters::attachment_payload_oversize,
    &QualityCounters::attachment_not_collected,
    &QualityCounters::attachment_publish_failed,
    &QualityCounters::sample_attempted,
    &QualityCounters::sample_failed,
    &QualityCounters::sample_skipped_busy,
    &QualityCounters::context_unavailable};

std::uint64_t monotonic_time(std::uint64_t producer) {
  auto value = timespec{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    const auto error = errno;
    throw CollectionError(CollectionErrorCode::Transfer, producer, "xpool cannot read collection clock", std::nullopt,
                          std::strerror(error));
  }
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(value.tv_nsec);
}

std::uint64_t saturated_add(std::uint64_t left, std::uint64_t right) {
  return right > std::numeric_limits<std::uint64_t>::max() - left ? std::numeric_limits<std::uint64_t>::max()
                                                                  : left + right;
}

std::uint64_t consume_id(std::uint64_t producer) {
  static auto identities = AttemptSequence{};
  const auto identity = identities.try_next();
  if (identity == 0) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, producer,
                          "xpool process consume-slot identities exhausted");
  }
  return identity;
}

} // namespace

Collector::Collector(HostBatchPool &batches, std::span<Event> host_scratch)
    : source_(batches.source()), batches_(batches), scratch_(host_scratch), consume_id_(consume_id(source_.producer)) {
  previous_.run_id = source_.run_id;
  previous_.producer_id = source_.producer;
}

std::size_t Collector::collect_host(HostBuffer &buffer, BatchInfo info, MetadataResolver resolver,
                                    ReconstructionSink reconstruction) {
  if (source_closed() || scratch_.size() < batches_.events_per_batch())
    throw CollectionError(CollectionErrorCode::ResourceBudget, source_.producer,
                          "xpool Host collection needs a live source and independent fixed scratch");
  const auto &origin = buffer.source().source();
  if (origin.producer != source_.producer || origin.clock_domain_id != source_.clock_domain_id ||
      origin.run_id != source_.run_id || info.metadata_revision == 0)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer, "xpool Host source mismatch");
  if (stop_requested())
    buffer.stop();
  const auto begin = monotonic_time(source_.producer);
  auto count = std::size_t{0};
  for (; count < batches_.events_per_batch(); ++count) {
    const auto read = buffer.source().try_acquire();
    if (!read.acquired)
      break;
    scratch_[count] = *read.event;
    if (buffer.source().release(read.token) != ReleaseStatus::Released) {
      fail_transfer();
      throw CollectionError(CollectionErrorCode::Transfer, source_.producer, "xpool Host copy ownership failed");
    }
  }
  observe_source(buffer.progress());
  if (count == 0)
    return 0;
  record_copied(count);
  const auto events = scratch_.first(count);
  try {
    std::ranges::sort(events, {}, &Event::event_sequence);
    if (resolver.freeze)
      info = resolver.freeze(resolver.context, events);
    if (reconstruction.merge)
      reconstruction.merge(reconstruction.context, events);
    info.host_window_begin_ns = begin;
    info.host_window_end_ns = monotonic_time(source_.producer);
  } catch (const CollectionError &cause) {
    record_unstaged_loss(count);
    buffer.stop();
    if (cause.code() == CollectionErrorCode::Transfer) {
      fail_transfer();
      throw;
    }
    fail(StopReason::MetadataUnavailable);
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool completed Host copy has no valid metadata", std::nullopt, cause.what());
  } catch (const std::exception &cause) {
    record_unstaged_loss(count);
    fail(StopReason::MetadataUnavailable);
    buffer.stop();
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer,
                          "xpool completed Host copy has no valid metadata", std::nullopt, cause.what());
  }
  const auto batch = export_failed_.load(std::memory_order_acquire) ? BatchReservation{} : batches_.try_reserve();
  if (batch.status != BatchReserveStatus::Reserved) {
    record_unstaged_loss(count);
    if (batch.status == BatchReserveStatus::IdentityExhausted) {
      fail(StopReason::IdentityExhausted);
      buffer.stop();
    }
    return count;
  }
  std::ranges::copy(events, batch.events.begin());
  try {
    if (batches_.publish(batch.token, count, info) != BatchPublishStatus::Published)
      throw std::logic_error("xpool Host batch owner rejected publication");
  } catch (const std::exception &cause) {
    record_unstaged_loss(count);
    batches_.cancel(batch.token);
    fail(StopReason::MetadataUnavailable);
    buffer.stop();
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source_.producer, "xpool Host handoff failed",
                          std::nullopt, cause.what());
  }
  return count;
}

void Collector::record_copied(std::uint64_t count) {
  if (source_closed()) {
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool source copy accounting is already retired");
  }
  copied_.add(count);
}

void Collector::record_unstaged_loss(std::uint64_t count) {
  if (source_closed()) {
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool source discard accounting is already retired");
  }
  lost_.add(count);
}

void Collector::observe_source(SourceProgress progress) {
  if (source_closed()) {
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool collector cannot update a sealed source image");
  }
  const auto &counts = progress.counts;
  const std::uint64_t values[] = {counts.attempted, counts.committed,   counts.dropped_full,
                                  counts.abandoned, counts.stale_token, counts.duplicate_commit};
  for (auto index = std::size_t{0}; index < publisher_max_.size(); ++index) {
    publisher_max_[index] = std::max(publisher_max_[index], values[index]);
    progress_[index].store(publisher_max_[index], std::memory_order_relaxed);
  }
  const auto flags = (counts.stopped ? source_stopped : 0U) | (counts.identity_exhausted ? source_exhausted : 0U) |
                     (counts.counter_exhausted ? source_saturated : 0U);
  source_flags_.fetch_or(flags, std::memory_order_release);
}

void Collector::seal_source(SourceCounts counts) {
  if (source_closed() || !counts.stopped) {
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool source finalization requires closed admission and one immutable image");
  }
  if (counts.exact_totals) {
    auto remaining = counts.attempted;
    const std::uint64_t outcomes[] = {counts.committed, counts.dropped_full, counts.abandoned, counts.pending};
    auto consistent = !counts.counter_exhausted && counts.last_attempted_sequence == counts.attempted;
    for (const auto value : outcomes) {
      if (value > remaining) {
        consistent = false;
        break;
      }
      remaining -= value;
    }
    if (!consistent || remaining != 0) {
      throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                            "xpool final source image violates attempt conservation");
    }
  }
  observe_source({.counts = counts, .pending_known = true});
  final_source_ = counts;
  source_closed_.store(true, std::memory_order_release);
  batches_.stop();
}

bool Collector::current(ConsumeToken token) const noexcept {
  return active_ && held_.acquired && token.batch == consume_id_ && token.generation == generation_;
}

[[noreturn]] void Collector::invalid_release(const char *cause, bool stale) const {
  if (stale && !export_closed_.load(std::memory_order_acquire)) {
    invalid_consumption_.add(1);
  }
  throw CollectionError(CollectionErrorCode::InvalidRelease, source_.producer,
                        "xpool collector ownership operation rejected", std::nullopt, cause);
}

PollResult poll(Collector &collector, std::uint32_t max_events) {
  if (max_events == 0 || collector.active_) {
    return {};
  }
  if (collector.generation_ == std::numeric_limits<std::uint64_t>::max()) {
    collector.fail(StopReason::IdentityExhausted);
    throw CollectionError(CollectionErrorCode::ResourceBudget, collector.source_.producer,
                          "xpool consume identity range exhausted");
  }
  if (!collector.held_.acquired) {
    collector.held_ = collector.batches_.try_acquire();
    if (!collector.held_.acquired) {
      return {};
    }
    collector.offset_ = 0;
    collector.accumulated_ = {};
  }
  collector.issued_ = std::min<std::size_t>(max_events, collector.held_.view.events.size() - collector.offset_);
  collector.active_ = true;
  collector.reported_ = false;
  ++collector.generation_;
  return {.token = ConsumeToken{.batch = collector.consume_id_, .generation = collector.generation_},
          .events = collector.held_.view.events.subspan(collector.offset_, collector.issued_)};
}

void Collector::report(ConsumeToken token, BatchOutcome outcome) {
  if (!current(token) || reported_) {
    invalid_release("report requires one active unreported prefix", true);
  }
  if (outcome.exported > issued_ || outcome.post_commit_lost != issued_ - outcome.exported) {
    invalid_release("report must cover exactly the issued prefix");
  }
  accumulated_.exported += outcome.exported;
  accumulated_.post_commit_lost += outcome.post_commit_lost;
  exported_.add(outcome.exported);
  lost_.add(outcome.post_commit_lost);
  reported_ = true;
}

BatchView Collector::view(ConsumeToken token) const {
  if (!current(token)) {
    invalid_release("view requires the current prefix identity", true);
  }
  auto result = held_.view;
  result.events = result.events.subspan(offset_, issued_);
  return result;
}

std::uint64_t Collector::discard_ready() {
  if (active_) {
    invalid_release("discard requires retired public payload reads");
  }
  auto count = std::uint64_t{0};
  for (auto index = std::size_t{0}; index < batches_.batch_capacity(); ++index) {
    if (!held_.acquired) {
      held_ = batches_.try_acquire();
      offset_ = 0;
      accumulated_ = {};
    }
    if (!held_.acquired) {
      break;
    }
    const auto remaining = held_.view.events.size() - offset_;
    accumulated_.post_commit_lost += remaining;
    lost_.add(remaining);
    count += remaining;
    if (batches_.report(held_.token, accumulated_) != BatchReportStatus::Recorded ||
        batches_.release(held_.token) != BatchReleaseStatus::Released) {
      invalid_release("discard could not retire its acquired generation");
    }
    held_ = {};
  }
  return count;
}

void release(Collector &collector, ConsumeToken token) {
  if (!collector.current(token)) {
    collector.invalid_release("release requires the current prefix identity", true);
  }
  if (!collector.reported_) {
    collector.invalid_release("release requires a reported prefix and retired reads");
  }
  collector.offset_ += collector.issued_;
  collector.active_ = false;
  collector.reported_ = false;
  if (collector.offset_ == collector.held_.view.events.size()) {
    const auto recorded = collector.batches_.report(collector.held_.token, collector.accumulated_);
    if (recorded != BatchReportStatus::Recorded ||
        collector.batches_.release(collector.held_.token) != BatchReleaseStatus::Released) {
      collector.invalid_release("underlying host batch did not accept its complete outcome");
    }
    collector.held_ = {};
  }
}

void Collector::finish_export() {
  if (!source_closed() || active_ || held_.acquired) {
    invalid_release("export retirement requires a closed source and retired leases");
  }
  held_ = batches_.try_acquire();
  if (held_.acquired) {
    offset_ = 0;
    accumulated_ = {};
    invalid_release("ready batches still require publication or counted discard");
  }
  const auto exported = exported_.quiescent();
  const auto lost = lost_.quiescent();
  if (exported.exhausted || lost.exhausted || exported.value > final_source_.committed ||
      lost.value != final_source_.committed - exported.value) {
    invalid_release("committed events still lack publication or counted discard");
  }
  export_closed_.store(true, std::memory_order_release);
}

void Collector::fail(StopReason reason) noexcept {
  auto expected = std::uint32_t{0};
  reason_.compare_exchange_strong(expected, static_cast<std::uint32_t>(reason) + 1, std::memory_order_release,
                                  std::memory_order_relaxed);
  if (reason == StopReason::DiskQuota || reason == StopReason::IoError || reason == StopReason::MetadataUnavailable) {
    export_failed_.store(true, std::memory_order_release);
  }
}

void Collector::fail_transfer() noexcept {
  fail(StopReason::IoError);
  transfer_uncertain_.store(true, std::memory_order_release);
}

void Collector::record(ExternalCounter counter, std::uint64_t count) {
  const auto index = static_cast<std::size_t>(counter);
  if (index >= external_.size() || export_closed_.load(std::memory_order_acquire)) {
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer,
                          "xpool external quality category is invalid or already retired");
  }
  external_[index].add(count);
}

void Collector::observe_fabric(FabricProgress progress) {
  const auto lock = std::lock_guard{quality_mutex_};
  if (source_closed())
    throw CollectionError(CollectionErrorCode::Transfer, source_.producer, "xpool Fabric counters are already sealed");
  fabric_progress_.operations = std::max(fabric_progress_.operations, progress.operations);
  fabric_progress_.dropped = std::max(fabric_progress_.dropped, progress.dropped);
  fabric_progress_.diagnostics |= progress.diagnostics;
  fabric_progress_.counter_exhausted |= progress.counter_exhausted;
  fabric_progress_.identity_exhausted |= progress.identity_exhausted;
  fabric_progress_.exact = progress.exact && !fabric_progress_.counter_exhausted;
}

FabricProgress Collector::fabric_progress() const {
  const auto lock = std::lock_guard{quality_mutex_};
  return fabric_progress_;
}

QualitySnapshot quality(const Collector &collector) {
  const auto lock = std::lock_guard{collector.quality_mutex_};
  auto result = collector.previous_;
  if (result.revision == std::numeric_limits<std::uint64_t>::max()) {
    throw CollectionError(CollectionErrorCode::ResourceBudget, collector.source_.producer,
                          "xpool quality image identity range exhausted");
  }
  ++result.revision;
  result.host_time_ns = monotonic_time(collector.source_.producer);
  const auto source_closed = collector.source_closed();
  const auto export_closed = collector.export_closed_.load(std::memory_order_acquire);
  const auto transfer_uncertain = collector.transfer_uncertain_.load(std::memory_order_acquire);
  const auto flags = collector.source_flags_.load(std::memory_order_acquire);
  const auto reason = collector.reason_.load(std::memory_order_acquire);
  const auto source_values = source_closed ? collector.final_source_ : SourceCounts{};
  const std::uint64_t final_values[] = {source_values.attempted,    source_values.committed,
                                        source_values.dropped_full, source_values.abandoned,
                                        source_values.stale_token,  source_values.duplicate_commit};
  std::uint64_t QualityCounters::*fields[] = {&QualityCounters::attempted,    &QualityCounters::committed,
                                              &QualityCounters::dropped_full, &QualityCounters::abandoned,
                                              &QualityCounters::stale_token,  &QualityCounters::duplicate_commit};
  for (auto index = std::size_t{0}; index < collector.progress_.size(); ++index) {
    const auto value = source_closed ? final_values[index] : collector.progress_[index].load(std::memory_order_relaxed);
    result.counters.*fields[index] = std::max(result.counters.*fields[index], value);
  }
  const auto copied = collector.copied_.lower_bound();
  const auto exported = collector.exported_.lower_bound();
  const auto lost = collector.lost_.lower_bound();
  const auto invalid = collector.invalid_consumption_.lower_bound();
  const auto source_stale =
      source_closed ? source_values.stale_token : collector.progress_[4].load(std::memory_order_relaxed);
  const auto stale_overflow = invalid.value > std::numeric_limits<std::uint64_t>::max() - source_stale;
  result.counters.stale_token = std::max(result.counters.stale_token, saturated_add(source_stale, invalid.value));
  result.counters.copied = std::max(result.counters.copied, copied.value);
  result.counters.exported = std::max(result.counters.exported, exported.value);
  result.counters.post_commit_lost = std::max(result.counters.post_commit_lost, lost.value);
  result.counters.copied = std::max(result.counters.copied, result.counters.exported);
  result.counter_exhausted = result.counter_exhausted || (flags & source_saturated) != 0 || copied.exhausted ||
                             exported.exhausted || lost.exhausted || invalid.exhausted || stale_overflow;
  for (auto index = std::size_t{0}; index < external_count; ++index) {
    const auto value = collector.external_[index].lower_bound();
    const auto fabric = index == 0   ? collector.fabric_progress_.operations
                        : index == 1 ? collector.fabric_progress_.dropped
                                     : 0;
    result.counters.*external_fields[index] =
        std::max(result.counters.*external_fields[index], saturated_add(value.value, fabric));
    result.counter_exhausted =
        result.counter_exhausted || fabric > std::numeric_limits<std::uint64_t>::max() - value.value;
    result.counter_exhausted = result.counter_exhausted || value.exhausted;
  }
  result.counter_exhausted = result.counter_exhausted || collector.fabric_progress_.counter_exhausted;
  // Distinct later-stage identities are also committed evidence, even when a
  // producer was preempted between publication and its cumulative add.
  result.counters.committed = std::max({result.counters.committed, result.counters.copied,
                                        saturated_add(result.counters.exported, result.counters.post_commit_lost)});
  result.counters.attempted = std::max(
      result.counters.attempted,
      saturated_add(saturated_add(result.counters.committed, result.counters.dropped_full), result.counters.abandoned));
  result.last_attempted_sequence = result.counters.attempted;
  result.collection_state = source_closed
                                ? CollectionState::Closed
                                : ((flags & source_stopped) != 0 ? CollectionState::Stopped : CollectionState::Active);
  result.export_state = collector.export_failed_.load(std::memory_order_acquire)
                            ? ExportState::Failed
                            : (export_closed ? ExportState::Closed : ExportState::Active);
  if (reason != 0 && reason <= static_cast<std::uint32_t>(StopReason::ShutdownDeadline) + 1) {
    result.stop_reason = static_cast<StopReason>(reason - 1);
  } else if ((flags & source_exhausted) != 0) {
    result.stop_reason = StopReason::IdentityExhausted;
  }
  result.pending =
      source_closed && !transfer_uncertain ? std::optional<std::uint64_t>{source_values.pending} : std::nullopt;
  result.exact_totals = source_closed && export_closed && source_values.exact_totals && !result.counter_exhausted &&
                        result.stop_reason != StopReason::MetadataUnavailable && !transfer_uncertain;
  if (result.exact_totals) {
    auto remaining = result.counters.attempted;
    const std::uint64_t accounted[] = {result.counters.committed, result.counters.dropped_full,
                                       result.counters.abandoned, source_values.pending};
    for (const auto value : accounted) {
      if (value > remaining) {
        result.exact_totals = false;
        break;
      }
      remaining -= value;
    }
    result.exact_totals = result.exact_totals && remaining == 0;
  }
  result.tail = source_closed && export_closed && !transfer_uncertain
                    ? (source_values.pending == 0 ? Tail::Complete : Tail::Pending)
                    : Tail::Unknown;
  collector.previous_ = result;
  return result;
}

} // namespace xpool::devkit::timeline
