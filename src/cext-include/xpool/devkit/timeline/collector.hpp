#pragma once

/// \file xpool/devkit/timeline/collector.hpp
/// \brief Bounded per-Producer consumption, outcome reporting and C0 quality.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>

#include <xpool/devkit/timeline/batch.hpp>
#include <xpool/devkit/timeline/collection.hpp>
#include <xpool/devkit/timeline/fabricrecording.hpp>
#include <xpool/devkit/timeline/handoff.hpp>
#include <xpool/devkit/timeline/quality.hpp>
#include <xpool/devkit/timeline/sink.hpp>

namespace xpool::devkit::timeline {

/// An immutable Host-owned prefix remains valid until its matching release.
struct PollResult {
  /// Empty when no ownership is granted, including a still-active prior lease.
  std::optional<ConsumeToken> token = std::nullopt;
  /// At most max_events values; empty results grant no payload access.
  std::span<const Event> events{};
};

/// Per-Producer handle above one source-bound Host batch pool. The allocation
/// owner keeps the handle, pool and all source storage alive through every
/// worker and lease. One collection lane publishes progress/copies; one Writer
/// lane serializes poll/report/release. quality calls serialize independently
/// and never hold a collection or file-I/O lock.
class Collector {
public:
  /// Bind a registered pool. Host collection additionally borrows an independent
  /// scratch span, disjoint from source/batch payload, of events_per_batch size.
  /// No allocation
  /// or worker starts here; the process service owns those lifetimes.
  /// \throws CollectionError if process consume-slot identities exhaust.
  explicit Collector(HostBatchPool &batches, std::span<Event> host_scratch = {});
  Collector(const Collector &) = delete;
  Collector &operator=(const Collector &) = delete;

  /// Fixed source; provenance is never inferred from Event fields.
  const BatchSource &source() const noexcept { return source_; }
  /// Collection-lane access for Host/Device copy drivers, never the Writer.
  HostBatchPool &batches() noexcept { return batches_; }
  /// Copy at most one bounded batch from a matching Host source. Complete
  /// payload copies retire source slots before Writer publication. Full staging
  /// drains into explicit post-commit loss; no queue grows or waits for I/O.
  /// With a resolver, freeze revision/hash after the copy completes. Otherwise
  /// info is an already validated immutable revision covering all source refs.
  /// \throws CollectionError for mismatched registration/metadata or copy state.
  std::size_t collect_host(HostBuffer &buffer, BatchInfo info, MetadataResolver resolver = {},
                           ReconstructionSink reconstruction = {});
  /// Collection lane records one completed copy per distinct Event identity.
  /// \throws CollectionError if source accounting has already retired.
  void record_copied(std::uint64_t count);
  /// Collection lane records known committed losses before Host batch admission.
  /// \throws CollectionError if source accounting has already retired.
  void record_unstaged_loss(std::uint64_t count);
  /// Collection lane publishes cumulative lower bounds; pending remains unknown.
  /// \throws CollectionError if a final source image was already sealed.
  void observe_source(SourceProgress progress);
  /// Publish one immutable final source image after admission closes and every
  /// source API caller retires. This alone does not finalize export or files.
  /// \throws CollectionError for repeated or unclosed source finalization.
  void seal_source(SourceCounts counts);
  /// Whether the collection lane has published its final source image.
  bool source_closed() const noexcept { return source_closed_.load(std::memory_order_acquire); }
  /// Writer records one real publication/discard result for the current prefix.
  /// \throws CollectionError with invalid_release for stale/repeated reports,
  /// or outcomes not covering exactly the issued prefix.
  void report(ConsumeToken token, BatchOutcome outcome);
  /// Frozen source and metadata for the active prefix, never a live revision.
  /// \throws CollectionError with invalid_release for another/stale token.
  BatchView view(ConsumeToken token) const;
  /// Writer discards ready data after a known failure, without allocating new
  /// lease identities. Active reads must retire first. At most pool capacity
  /// batches are inspected; later copies are handled on the next tick.
  /// \throws CollectionError if a public lease is still active.
  std::uint64_t discard_ready();
  /// Writer marks retirement only after collection closes and all queued/held
  /// batches retire. Failed export remains failed in the final quality image.
  /// \throws CollectionError if ownership or ready batches still remain.
  void finish_export();
  /// Record a failure and request source stop. The process driver performs the
  /// actual Host/Device stop, then publishes its observed source state.
  void fail(StopReason reason) noexcept;
  /// Mark transfer completion uncertain and request io_error. Preserve an
  /// earlier stop reason; uncertainty independently prevents exact totals,
  /// known pending or a complete tail for every subsequent quality image.
  void fail_transfer() noexcept;
  /// Whether the source driver must close admission and drain known losses.
  bool stop_requested() const noexcept { return reason_.load(std::memory_order_acquire) != 0; }
  /// Record independent view/attachment/sample facts, never Event attempts.
  /// \throws CollectionError for an invalid category or updates after export retirement.
  void record(ExternalCounter counter, std::uint64_t count = 1);
  /// Consume a completed Fabric D2H sample once. Absolute totals retain maxima;
  /// they are independent of Event attempts and other legacy domains.
  void observe_fabric(FabricProgress progress);
  /// Copy bounded diagnostics/counts; exact is meaningful only while the caller
  /// holds the matching production quiescence used for its fresh drain.
  FabricProgress fabric_progress() const;

private:
  friend PollResult poll(Collector &collector, std::uint32_t max_events);
  friend void release(Collector &collector, ConsumeToken token);
  friend QualitySnapshot quality(const Collector &collector);
  bool current(ConsumeToken token) const noexcept;
  [[noreturn]] void invalid_release(const char *cause, bool stale = false) const;

  const BatchSource source_;
  HostBatchPool &batches_;
  const std::span<Event> scratch_;
  const std::uint64_t consume_id_;
  BatchAcquisition held_{};
  std::size_t offset_ = 0;
  std::size_t issued_ = 0;
  std::uint64_t generation_ = 0;
  bool active_ = false;
  bool reported_ = false;
  BatchOutcome accumulated_{};
  ConcurrentCounter copied_{};
  ConcurrentCounter exported_{};
  ConcurrentCounter lost_{};
  mutable ConcurrentCounter invalid_consumption_{};
  std::array<ConcurrentCounter, static_cast<std::size_t>(ExternalCounter::Count)> external_{};
  std::array<std::atomic<std::uint64_t>, 6> progress_{};
  std::array<std::uint64_t, 6> publisher_max_{};
  std::atomic<std::uint32_t> source_flags_{0};
  SourceCounts final_source_{};
  std::atomic<bool> source_closed_{false};
  std::atomic<bool> export_closed_{false};
  std::atomic<bool> export_failed_{false};
  std::atomic<bool> transfer_uncertain_{false};
  std::atomic<std::uint32_t> reason_{0};
  mutable std::mutex quality_mutex_;
  mutable QualitySnapshot previous_{};
  FabricProgress fabric_progress_{};
};

/// Writer leases at most max_events. A prior active token returns an empty
/// result; a smaller request preserves the unreturned suffix for later polls.
/// \throws CollectionError with resource_budget at lease identity exhaustion.
PollResult poll(Collector &collector, std::uint32_t max_events);
/// Retire a prefix after its outcome is reported and every payload read ends.
/// \throws CollectionError with invalid_release for stale, double, or unreported
/// release. Failure retains ownership; release never implies file publication.
void release(Collector &collector, ConsumeToken token);
/// Return cumulative typed C0 evidence. Active images contain monotone lower
/// bounds and unknown pending; exact final values require source/export
/// retirement and nonsaturated counters. No producer or Writer is frozen.
/// \throws CollectionError if quality-image identities exhaust or the Host clock fails.
QualitySnapshot quality(const Collector &collector);

} // namespace xpool::devkit::timeline
