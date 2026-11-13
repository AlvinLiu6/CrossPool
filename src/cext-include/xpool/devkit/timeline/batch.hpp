#pragma once

/// \file xpool/devkit/timeline/batch.hpp
/// \brief Source-bound Host batches and exactly-once export outcome accounting.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <xpool/devkit/timeline/chunk.hpp>
#include <xpool/devkit/timeline/counter.hpp>
#include <xpool/devkit/timeline/source.hpp>

namespace xpool::devkit::timeline {

/// Immutable provenance frozen when a completed copy is published to a Writer.
struct BatchInfo {
  /// Positive revision resolving all references in this batch.
  std::uint64_t metadata_revision = 0;
  /// First 16 bytes of SHA-256 over the exact immutable metadata file.
  std::array<std::byte, 16> metadata_sha256_prefix{};
  /// CLOCK_MONOTONIC beginning of the collection window, in nanoseconds.
  std::uint64_t host_window_begin_ns = 0;
  /// CLOCK_MONOTONIC end of the collection window, at least its beginning.
  std::uint64_t host_window_end_ns = 0;
};

/// Authority over one fill generation in its originating pool.
struct BatchWriteToken {
  /// Zero-based batch slot, never an Event identity.
  std::uint64_t batch = 0;
  /// Positive private generation of this reservation.
  std::uint64_t generation = 0;
};

/// Authority over one acquired immutable generation in its originating pool.
struct BatchReadToken {
  /// Zero-based batch slot granted by acquisition.
  std::uint64_t batch = 0;
  /// Generation held until outcome accounting and all payload reads retire.
  std::uint64_t generation = 0;
};

/// Host capacity status; Full does not allocate an attempted-event sequence.
enum class BatchReserveStatus : std::uint32_t { Reserved, Full, Stopped, IdentityExhausted };
/// Publication grants read eligibility once; rejected tokens change no batch.
enum class BatchPublishStatus : std::uint32_t { Published, StaleToken, AlreadyPublished };
/// Cancellation retires an unused fill reservation, not committed Device facts.
enum class BatchCancelStatus : std::uint32_t { Canceled, StaleToken, AlreadyPublished };
/// An acquired generation can contribute one complete export outcome.
enum class BatchReportStatus : std::uint32_t { Recorded, StaleToken, AlreadyRecorded, NotAcquired };
/// Release requires a previously recorded outcome and completed payload reads.
enum class BatchReleaseStatus : std::uint32_t { Released, StaleToken, OutcomeMissing, NotAcquired };

/// Fill storage is writable only while the returned reservation remains owned.
struct BatchReservation {
  /// Only Reserved grants ownership; other results carry zero/empty fields.
  BatchReserveStatus status = BatchReserveStatus::Stopped;
  /// Generation-scoped authority; copying it creates no additional owner.
  BatchWriteToken token{};
  /// Fixed-capacity fill area; retire every write before publish or cancel.
  std::span<Event> events{};
};

/// Complete declared outcome after publication or a definite counted discard.
struct BatchOutcome {
  /// Distinct Events successfully published, not merely encoded or written.
  std::uint64_t exported = 0;
  /// Distinct copied Events definitively discarded after commit.
  std::uint64_t post_commit_lost = 0;
};

/// Immutable Writer input. Events and any derived Chunk remain valid only
/// until release of the originating read token. Metadata must be published
/// before a referencing file; this view preserves its exact revision/hash.
struct BatchView {
  /// Registered source copied from the originating pool.
  BatchSource source{};
  /// Provenance captured when the copy completed, never a later live revision.
  BatchInfo info{};
  /// Immutable committed Host payload owned by this read generation.
  std::span<const Event> events{};

  /// Construct encoding input using the Writer's positive chunk sequence.
  /// This performs no publication or accounting. The Chunk borrows these Events.
  /// \throws std::invalid_argument when chunk_sequence is zero.
  Chunk chunk(std::uint64_t chunk_sequence) const;
};

/// An empty acquisition grants neither a token nor payload access.
struct BatchAcquisition {
  /// Whether the result holds one immutable batch generation.
  bool acquired = false;
  /// Writer ownership, distinct from Collector fill authority.
  BatchReadToken token{};
  /// Source, frozen metadata and Events held until explicit release.
  BatchView view{};
};

/// Borrowed Host control; callers initialize it and leave it to the pool.
struct alignas(8) BatchControl {
  /// Four phase bits leave this positive private generation range.
  static constexpr std::uint64_t generation_limit = (std::uint64_t{1} << 60) - 1;
  /// Packed generation/phase. First generation starts Free.
  std::uint64_t ownership = std::uint64_t{1} << 4;
  /// Published Event count, read only while holding this generation.
  std::size_t event_count = 0;
  /// Immutable metadata and window installed before ready publication.
  BatchInfo info{};
};

/// Borrowed scheduling/counters. Initialize to zero; subsequent direct access
/// requires both lanes to be quiescent. Each counter has one writing lane.
struct alignas(8) BatchPoolState {
  /// Collector-owned scan origin, unrelated to event or chunk sequences.
  std::size_t write_cursor = 0;
  /// Writer-owned scan origin; neither lane waits for the other.
  std::size_t read_cursor = 0;
  /// Private admission stop flags, accessed atomically during service.
  std::uint64_t stopped = 0;
  /// Collector-owned count of completed copies accepted by this pool.
  SaturatingCounter copied;
  /// Writer-owned count of Events reported successfully published.
  SaturatingCounter exported;
  /// Writer-owned count of Events reported definitively discarded.
  SaturatingCounter post_commit_lost;
};

/// Actual single-allocation geometry for this pool, not process admission.
struct BatchPoolLayout {
  /// Event payload slab bytes, beginning at allocation offset zero.
  std::size_t event_bytes;
  /// Aligned beginning of per-batch controls.
  std::size_t control_offset;
  /// Complete control storage derived from BatchControl.
  std::size_t control_bytes;
  /// Aligned beginning of lane state and counters.
  std::size_t state_offset;
  /// Actual BatchPoolState bytes.
  std::size_t state_bytes;
  /// Complete allocation bytes including final alignment rounding.
  std::size_t total_bytes;
};

/// Quiescent occupancy and export-stage totals for this pool. This is neither
/// a live checkpoint nor complete source quality or legacy snapshot diagnostics.
struct BatchCounts {
  /// Batches available to the Collector.
  std::size_t free = 0;
  /// Reservations still owned by the Collector.
  std::size_t filling = 0;
  /// Published batches awaiting acquisition.
  std::size_t ready = 0;
  /// Writer-owned batches awaiting an outcome.
  std::size_t reading = 0;
  /// Accounted batches still held by the Writer.
  std::size_t finalized = 0;
  /// Batches retired at private generation exhaustion.
  std::size_t retired = 0;
  /// Cumulative completed Event copies accepted by the pool.
  std::uint64_t copied = 0;
  /// Cumulative Events reported published.
  std::uint64_t exported = 0;
  /// Cumulative copied Events reported lost.
  std::uint64_t post_commit_lost = 0;
  /// Whether any total is saturated lower-bound evidence.
  bool counter_exhausted = false;
  /// Whether new reservation admission has stopped.
  bool stopped = false;
  /// Whether private generation exhaustion stopped admission.
  bool identity_exhausted = false;
};

/// Fixed-capacity single-Collector/single-Writer Host handoff. Each lane
/// serializes its calls; the two lanes may execute concurrently and hold
/// multiple batches. Collector submits each committed Event identity once.
/// Tokens and borrowed storage belong to this pool for its entire lifetime.
/// Methods perform no I/O, allocation, mutex wait or unbounded retry.
class HostBatchPool {
public:
  /// Borrow disjoint payload/control/state and bind one registered source.
  /// \pre Construction is exclusive, controls are Free with positive generations
  /// and zero counts, state is zero, and all storage outlives its owners.
  /// \throws std::invalid_argument for invalid source, geometry or initial state;
  /// std::overflow_error for arithmetic overflow. The future Collector facade
  /// translates these local errors into its context-bearing CollectionError.
  HostBatchPool(BatchSource source, std::size_t events_per_batch, std::span<Event> events,
                std::span<BatchControl> batches, BatchPoolState &state);

  HostBatchPool(const HostBatchPool &) = delete;
  HostBatchPool &operator=(const HostBatchPool &) = delete;

  /// Collector scans each batch at most once and claims one Free generation.
  /// Full means each candidate was occupied at its own claim point, not one
  /// globally atomic fullness snapshot. Failed results grant no ownership.
  BatchReservation try_reserve();

  /// Collector publishes a completed immutable copy and freezes its provenance.
  /// \pre All writes, including asynchronous transfers, have completed.
  /// \throws std::invalid_argument for zero/oversized count, invalid metadata
  /// or window, wrong source clock or zero/nonincreasing sequences. Invalid
  /// payload/provenance keeps the fill reservation available for correction.
  BatchPublishStatus publish(BatchWriteToken token, std::size_t event_count, BatchInfo info);

  /// Collector retires an unused reservation after every writer has retired.
  /// This records no copied/lost Events; source-side unstaged loss is separate.
  BatchCancelStatus cancel(BatchWriteToken token);

  /// Writer acquire-observes one ready batch, skipping unpublished reservations.
  BatchAcquisition try_acquire();

  /// Writer records one complete outcome and retains immutable read ownership.
  /// \pre Publication or a definite discard has been established externally.
  /// \throws std::invalid_argument unless exported + post_commit_lost equals
  /// the batch count; failure leaves ownership and counters unchanged.
  BatchReportStatus report(BatchReadToken token, BatchOutcome outcome);

  /// Writer recycles only an accounted generation after all payload reads end.
  /// OutcomeMissing leaves storage held. Timeout alone never permits release.
  BatchReleaseStatus release(BatchReadToken token);

  /// Stop admission; already-started admissions and existing owners may retire.
  /// May be called by shutdown concurrently with either lane.
  void stop();

  /// Inspect only after both lanes' API calls quiesce. Held fill/read ownership
  /// may remain. Totals cover accepted Host batches, not uncopied Device losses.
  BatchCounts quiescent_counts() const;

  /// Immutable source bound before either lane begins service.
  const BatchSource &source() const noexcept { return source_; }
  /// Fixed upper bound for one completed Host batch.
  std::size_t events_per_batch() const noexcept { return events_per_batch_; }
  /// Maximum batches a bounded Writer drain needs to inspect.
  std::size_t batch_capacity() const noexcept { return batches_.size(); }

  /// Checked payload/control/state geometry for one fixed Host allocation.
  /// \throws std::invalid_argument for invalid dimensions/alignment and
  /// std::overflow_error for multiplication, offset or rounding overflow.
  static BatchPoolLayout layout(std::size_t batch_count, std::size_t events_per_batch,
                                std::size_t allocation_alignment = 256);

private:
  bool valid_token(std::uint64_t batch, std::uint64_t generation) const;
  void recycle(std::size_t batch, std::uint64_t generation);

  const BatchSource source_;
  const std::size_t events_per_batch_;
  std::span<Event> events_;
  std::span<BatchControl> batches_;
  BatchPoolState &state_;
};

} // namespace xpool::devkit::timeline
