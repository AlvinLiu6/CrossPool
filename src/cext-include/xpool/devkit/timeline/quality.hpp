#pragma once

/// \file xpool/devkit/timeline/quality.hpp
/// \brief Typed C0 source quality, independent stage counters and unknown tails.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <xpool/devkit/timeline/contract.hpp>

namespace xpool::devkit::timeline {

/// Source admission and ownership lifecycle.
enum class CollectionState : std::uint32_t { Active, Stopped, Closed };
/// Export lifecycle, independent of source admission.
enum class ExportState : std::uint32_t { Active, Failed, Closed };
/// Why collection stopped; absence is represented by an optional value.
enum class StopReason : std::uint32_t {
  DiskQuota,
  IoError,
  MetadataUnavailable,
  IdentityExhausted,
  ShutdownDeadline,
};
/// Whether accepted work has a known complete, pending, or unknown tail.
enum class Tail : std::uint32_t { Complete, Pending, Unknown };

/// Cumulative C0 counters. They are never summed across checkpoint revisions.
struct QualityCounters {
  /// Accepted reservation attempts, including Full.
  std::uint64_t attempted = 0;
  /// Distinct immutable commits.
  std::uint64_t committed = 0;
  /// Attempts rejected by event capacity.
  std::uint64_t dropped_full = 0;
  /// Reservations explicitly retired without commit.
  std::uint64_t abandoned = 0;
  /// Rejected token operations excluding duplicate commit claims.
  std::uint64_t stale_token = 0;
  /// Repeated commit claims for an existing generation.
  std::uint64_t duplicate_commit = 0;
  /// Distinct complete copies into Host staging.
  std::uint64_t copied = 0;
  /// Distinct Events whose complete raw file was published.
  std::uint64_t exported = 0;
  /// Distinct committed Events definitively discarded after commit.
  std::uint64_t post_commit_lost = 0;
  /// Original operations considered for bounded legacy views.
  std::uint64_t snapshot_operations_attempted = 0;
  /// Operations beyond the retained legacy-view capacity.
  std::uint64_t snapshot_operations_dropped = 0;
  /// Independent attachment identity attempts.
  std::uint64_t attachment_attempted = 0;
  /// Attachment reservations rejected by their capacity.
  std::uint64_t attachment_capacity_exhausted = 0;
  /// Attachments whose payload exceeds configured bounds.
  std::uint64_t attachment_payload_oversize = 0;
  /// Attachments not copied into collection storage.
  std::uint64_t attachment_not_collected = 0;
  /// Attachments whose complete publication failed.
  std::uint64_t attachment_publish_failed = 0;
  /// Raw clock-pair sampling attempts.
  std::uint64_t sample_attempted = 0;
  /// Failed raw sampling attempts.
  std::uint64_t sample_failed = 0;
  /// Sampling opportunities skipped because the single slot was busy.
  std::uint64_t sample_skipped_busy = 0;
  /// Events lacking observation Context association.
  std::uint64_t context_unavailable = 0;
};

/// Native representation of one immutable C0 quality image. Active lower-bound
/// images explicitly omit pending and never claim exact or final totals.
struct QualitySnapshot {
  /// Version shared with raw chunks and metadata.
  std::uint32_t contract_version = kContractVersion;
  /// Run UUID bytes in canonical order.
  std::array<std::byte, 16> run_id{};
  /// Producer represented by this per-source Collector handle.
  std::uint64_t producer_id = 0;
  /// Positive unique image identity within this Producer.
  std::uint64_t revision = 0;
  /// CLOCK_MONOTONIC observation time in nanoseconds.
  std::uint64_t host_time_ns = 0;
  /// Source admission lifecycle, independent of export failure.
  CollectionState collection_state = CollectionState::Active;
  /// File-export lifecycle.
  ExportState export_state = ExportState::Active;
  /// Present only after a known stop cause.
  std::optional<StopReason> stop_reason = std::nullopt;
  /// Last attempted Event identity, never a buffer cursor.
  std::uint64_t last_attempted_sequence = 0;
  /// Cumulative source, stage, view, attachment and sample evidence.
  QualityCounters counters{};
  /// Uncommitted reservations; unknown is null, never a fabricated zero.
  std::optional<std::uint64_t> pending = std::nullopt;
  /// Completeness of accepted work after the observed boundary.
  Tail tail = Tail::Unknown;
  /// Whether a counter exceeded its representable range.
  bool counter_exhausted = false;
  /// True only when the checkpoint's counters and pending count are coherent and exact.
  bool exact_totals = false;
};

/// Independent observations outside the Event and export lifecycle. Modules
/// update their own categories without inventing Event attempts or commits.
enum class ExternalCounter : std::size_t {
  SnapshotOperationsAttempted,
  SnapshotOperationsDropped,
  AttachmentAttempted,
  AttachmentCapacityExhausted,
  AttachmentPayloadOversize,
  AttachmentNotCollected,
  AttachmentPublishFailed,
  SampleAttempted,
  SampleFailed,
  SampleSkippedBusy,
  ContextUnavailable,
  Count,
};

} // namespace xpool::devkit::timeline
