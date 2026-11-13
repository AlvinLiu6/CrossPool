#pragma once

/// \file xpool/devkit/timeline/writer.hpp
/// \brief Bounded Native raw-file publication on the dedicated Writer lane.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <xpool/devkit/timeline/collector.hpp>
#include <xpool/devkit/timeline/disk.hpp>
#include <xpool/devkit/timeline/metadata.hpp>
#include <xpool/devkit/timeline/resource.hpp>

namespace xpool::devkit::timeline {

/// Immutable evidence returned only after metadata publication succeeds.
struct MetadataPublication {
  /// Positive revision resolving the associated source's Event references.
  std::uint64_t revision;
  /// Exact SHA-256 of the supplied UTF-8 file including its final newline.
  std::array<std::byte, 32> sha256;
};

/// One registered Producer's serial Writer. Run directories are owned by this
/// process; one shared DiskBudget covers every directory and Producer. The
/// caller accounts this control object and bounded filesystem path storage in
/// the process Host budget. Scratch and revision registry use that same budget.
/// File I/O runs on the Writer lane, never on a Producer or collection lane.
class RawWriter {
public:
  /// Allocate fixed scratch and bounded revision capacity before service. The
  /// budgets outlive this Writer. Registration/semantic metadata validation is
  /// performed by the owning registry before publish_metadata().
  /// \throws CollectionError for invalid sizing or memory admission/allocation.
  RawWriter(BatchSource source, std::filesystem::path outdir, MemoryBudget &host, DiskBudget &disk,
            std::size_t chunk_bytes = Chunk::maximum_bytes, std::size_t metadata_capacity = 64);
  /// Retire scratch after all serial Writer calls finish.
  ~RawWriter();
  RawWriter(const RawWriter &) = delete;
  RawWriter &operator=(const RawWriter &) = delete;

  /// Publish validated C0 metadata bytes before referencing chunks. References
  /// are immutable across revisions; the registry owns that semantic check.
  /// Every supplied file has a final newline. Capacity cannot grow at runtime.
  /// \throws CollectionError with metadata_unavailable, resource_budget or io;
  /// failed temporary extents remain present and charged to the disk ledger.
  MetadataPublication publish_metadata(std::uint64_t revision, std::span<const std::byte> bytes);
  /// Publish all currently registered complete revisions on the Writer lane.
  /// Used at final closure even if the last revision has no referencing chunk.
  /// \throws CollectionError for identity, capacity, quota or file I/O failure.
  void publish_pending_metadata(const MetadataCatalog &catalog);
  /// Encode and atomically publish one nonempty source-bound immutable batch.
  /// Success is returned only after close and no-replace atomic rename. No
  /// fallible operation follows successful publication within this call.
  /// \throws CollectionError; the caller reports a definite loss and stops the
  /// affected source. A failed temporary file is retained as evidence.
  void publish(BatchView batch);
  /// Consume at most one prefix, publish, report the actual result and release.
  /// Failure sets the typed stop reason, counts this prefix as lost and
  /// propagates the error to the process Writer. Other queued data is retained
  /// until a later bounded discard. The Collector belongs to the same source.
  /// \throws CollectionError for publication or ownership failures.
  bool write_one(Collector &collector, const MetadataCatalog *catalog = nullptr);
  /// Publish one complete immutable C0 quality image from the fixed typed
  /// fields, including nullable pending and cumulative u64 counters.
  /// \throws CollectionError with resource_budget, metadata_unavailable or io.
  void publish_quality(const QualitySnapshot &snapshot);
  /// Fixed chunk capacity, including its header.
  std::size_t chunk_bytes() const noexcept { return chunk_bytes_; }

private:
  void ensure_metadata(std::uint64_t revision, const MetadataCatalog &catalog);
  void begin_path(std::string_view prefix);
  void append_number(std::uint64_t value);
  void publish_file(std::span<const std::byte> bytes, DiskKind kind);
  const BatchSource source_;
  MemoryBudget &host_;
  DiskBudget &disk_;
  const std::size_t chunk_bytes_;
  const std::size_t metadata_capacity_;
  BudgetMemoryResource path_resource_;
  std::pmr::string outdir_;
  std::pmr::string final_path_;
  std::pmr::string temporary_path_;
  BudgetMemoryResource registry_resource_;
  std::pmr::vector<MetadataPublication> metadata_;
  std::byte *scratch_ = nullptr;
  AttemptSequence chunks_{};
  std::uint64_t quality_revision_ = 0;
};

} // namespace xpool::devkit::timeline
