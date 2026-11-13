#pragma once

/// \file xpool/devkit/timeline/snapshotwriter.hpp
/// \brief Quota-admitted, hash-paired legacy snapshot replacement.

#include <memory>
#include <memory_resource>
#include <string>
#include <string_view>

#include <xpool/devkit/timeline/disk.hpp>
#include <xpool/devkit/timeline/snapshot.hpp>

namespace xpool::devkit::timeline {

/// Legacy formatting owned by its existing schema adapter. The callback must
/// deterministically stream the supplied immutable read and stable registered
/// metadata in both passes, including the final newline. Its borrowed context
/// and any additional staging are admitted by the runtime before service.
struct SnapshotDocument {
  /// Registered legacy envelope metadata, alive for both synchronous passes.
  void *context = nullptr;
  /// Format exactly this read; never perform an independent Native getter.
  void (*serialize)(const SnapshotRead &, void *, JsonSink) = nullptr;
};

/// One lifecycle path, called serially on the Writer lane. Reports replace the
/// same sidecar before replacing their snapshot; readers verify the exact hash.
/// Both new temporary files coexist with old files at peak quota. Atomicity is
/// per file, not a transaction spanning the pair. Any failure stops this owner
/// and retains temporary evidence and conservative disk charges.
class SnapshotPublisher {
  class ConstructionKey {
    friend class SnapshotPublisher;
    ConstructionKey() = default;
  };

public:
  /// Register a unique fresh snapshot path and its `.evidence.json` sidecar.
  /// The parent is an existing process-owned directory. All four final/temp
  /// paths must be absent at registration; only this owner may replace them.
  /// Path buffers and the implementation-sized shared control block are charged.
  /// \throws CollectionError for admission, allocation, or path collisions.
  static std::shared_ptr<SnapshotPublisher> create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                                                   std::string_view path, detail::LegacyScope scope);
  SnapshotPublisher(const SnapshotPublisher &) = delete;
  SnapshotPublisher &operator=(const SnapshotPublisher &) = delete;
  /// Measure/hash without retaining JSON, admit both exact extents, publish the
  /// sidecar, then publish the snapshot. Rehash the actual snapshot write and
  /// reject a changing formatter before its rename. No old files are deleted.
  /// Unknown original operation totals cannot publish a legacy envelope.
  /// \throws CollectionError for missing metadata, quota or I/O; adapter
  /// exceptions propagate after the temporary extent is retained and charged.
  void publish(const SnapshotRead &read, SnapshotDocument document);
  /// True after any attempted publication fails; subsequent writes reject.
  bool failed() const noexcept { return failed_; }
  /// Fixed lifecycle destination, valid while this publisher remains alive.
  std::string_view path() const noexcept { return snapshot_; }
  /// Allocator-only constructor; use create() for complete control admission.
  SnapshotPublisher(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                    std::string_view path, detail::LegacyScope scope);

private:
  friend class SnapshotExport;
  friend class CollectionService;
  // Only a closed, drained mailbox can transfer this destination. Retaining
  // the same publisher preserves its old file extents until actual replacement.
  void rebind(detail::LegacyScope scope) noexcept { scope_ = scope; }
  const std::shared_ptr<MemoryBudget> host_;
  const std::shared_ptr<DiskBudget> disk_;
  detail::LegacyScope scope_;
  BudgetMemoryResource paths_;
  std::pmr::string snapshot_;
  std::pmr::string evidence_;
  std::pmr::string snapshot_temporary_;
  std::pmr::string evidence_temporary_;
  DiskBudget::Extent snapshot_extent_{};
  DiskBudget::Extent evidence_extent_{};
  bool failed_ = false;
};

} // namespace xpool::devkit::timeline
