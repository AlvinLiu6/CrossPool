#pragma once

/// \file xpool/devkit/timeline/snapshotexport.hpp
/// \brief One bounded immutable snapshot mailbox for the process Writer.

#include <memory>
#include <mutex>
#include <optional>

#include <xpool/devkit/timeline/legacyjson.hpp>
#include <xpool/devkit/timeline/writer.hpp>

namespace xpool::devkit::timeline {

/// Publication progress independent of read-memory retirement and Event counts.
struct SnapshotExportReport {
  /// Number of accepted immutable reads; busy rejections are excluded.
  std::uint64_t accepted = 0;
  /// Snapshot/sidecar pairs whose two replacements both succeeded.
  std::uint64_t published = 0;
  /// A read is queued or actively held by the Writer.
  bool pending = false;
  /// Admission closed after production/read callers retired.
  bool closed = false;
  /// Sticky publication failure; no subsequent submission is accepted.
  std::optional<CollectionErrorCode> error = std::nullopt;
};

/// Fixed one-read mailbox bound to a unique legacy lifecycle path. Registration
/// admits the formatter, path buffers and shared control through process ledgers.
/// The caller obtains reads only after original production quiescence and its
/// required collection cutoff. One serialized Writer owns write_one().
class SnapshotExport {
  class ConstructionKey {
    friend class SnapshotExport;
    ConstructionKey() = default;
  };

public:
  /// Register the whole publication path before this lifecycle emits evidence.
  /// \throws CollectionError on metadata, quota, path or Host admission failure.
  static std::shared_ptr<SnapshotExport> create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                                                std::string_view path, detail::LegacyScope scope, std::size_t capacity,
                                                std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata);
  SnapshotExport(const SnapshotExport &) = delete;
  SnapshotExport &operator=(const SnapshotExport &) = delete;
  /// Retain one already-materialized read. Uses try_lock and returns false for
  /// busy, closed or failed state; rejection preserves the caller's read. Never
  /// waits on file I/O, formats JSON, allocates or creates another snapshot.
  /// \throws CollectionError for a null/wrong-domain read or unknown counts.
  bool submit(std::shared_ptr<const SnapshotRead> read);
  /// Prepare a fresh lifecycle on this destination after close and complete
  /// Writer retirement. Preserve the publisher and its actual file extents;
  /// new formatter/control admission precedes the no-fail scope change. Old
  /// mailboxes remain closed. An unavailable final read does not poison an
  /// otherwise healthy file owner. Serialized lifecycle ownership prevents any
  /// subsequent write_one() using the old publisher scope.
  /// \throws CollectionError for busy/open/failed state, another Source/family,
  /// or Host admission. Failure leaves the previous destination intact.
  std::shared_ptr<SnapshotExport> replacement(detail::LegacyScope scope, std::size_t capacity,
                                              std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata);
  /// Publish at most one accepted read on the sole Writer lane. A service
  /// supplies its matching writer/catalog so complete metadata revisions are
  /// published before either pair member, including zero-Event lifecycles.
  /// The two optional arguments are supplied together; low-level standalone
  /// callers own their external metadata prerequisites. Exceptions mark this
  /// owner failed and close admission while preserving file evidence.
  /// \throws CollectionError from pair publication; other exceptions propagate.
  bool write_one(RawWriter *metadata_writer = nullptr, const MetadataCatalog *metadata = nullptr);
  /// Close future submission while keeping an accepted read drainable.
  void close();
  /// Record that a final read cannot be materialized, without claiming a new
  /// accepted/published pair or changing older files. A queued/active write or
  /// busy state returns false; its ownership must first retire. The caller may
  /// safely release GPU resources after an independent producer/copy cutoff.
  bool close_unavailable(CollectionErrorCode error);
  /// Match the immutable Source and complete observer lifetime.
  bool belongs_to(const detail::LegacyScope &scope) const noexcept;
  /// Copy bounded publication counters under the short state mutex.
  SnapshotExportReport report() const;
  /// Registered source identity for service error attribution.
  const BatchSource &source() const noexcept { return scope_.source; }
  /// Fixed destination used to reject duplicate process registrations.
  std::string_view path() const noexcept { return publisher_->path(); }
  /// Allocator-only constructor; use create() for complete admission.
  SnapshotExport(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                 std::string_view path, detail::LegacyScope scope, std::size_t capacity,
                 std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata,
                 std::shared_ptr<SnapshotPublisher> publisher = {});

private:
  friend class CollectionService;
  const std::shared_ptr<MemoryBudget> host_;
  const detail::LegacyScope scope_;
  const std::size_t capacity_;
  const std::shared_ptr<LegacyJsonFormatter> formatter_;
  const std::shared_ptr<SnapshotPublisher> publisher_;
  mutable std::mutex mutex_;
  std::shared_ptr<const SnapshotRead> pending_;
  SnapshotExportReport report_{};
  bool replaced_ = false;
};

} // namespace xpool::devkit::timeline
