#pragma once

/// \file xpool/devkit/timeline/metadata.hpp
/// \brief Fixed-capacity immutable metadata revisions and post-copy resolution.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

#include <xpool/devkit/timeline/batch.hpp>
#include <xpool/devkit/timeline/handoff.hpp>
#include <xpool/devkit/timeline/resource.hpp>

namespace xpool::devkit::timeline {

/// Borrowed immutable revision bytes. The catalog and its Host budget remain
/// alive through every reader; published slots are never modified or reused.
struct MetadataRevision {
  /// Registered source shared by the file and referencing chunks.
  BatchSource source;
  /// Positive increasing document revision, starting at one.
  std::uint64_t revision;
  /// SHA-256 of exact UTF-8 bytes including the final newline.
  std::array<std::byte, 32> sha256;
  /// Exact validated document, retained in preallocated storage.
  std::span<const std::byte> bytes;
};

/// Optional retained semantic registry for a catalog. Install once during cold
/// preparation, before any collection call. Validation runs after the complete
/// immutable copy and checks its exact selected revision without allocation or
/// file I/O. The owner remains alive through every catalog/collection reader.
struct MetadataCoverage {
  /// Retained registry state; its catalog reference is weak to avoid a cycle.
  std::shared_ptr<void> owner;
  /// Throw CollectionError if any copied reference is absent in that revision.
  void (*validate)(void *, std::span<const Event>, std::uint64_t) = nullptr;
};

/// Per-Producer append-only revision storage. A cold registration owner first
/// validates full C0 metadata, unchanged prior definitions and all references,
/// then appends its complete snapshot before allowing Events with new references.
/// This byte/ownership layer does not parse or validate the JSON schema itself.
/// All revision slots and maximum payload bytes are admitted before service.
/// Exhaustion fails explicitly; there is no heap fallback, eviction or truncation.
class MetadataCatalog {
  class ConstructionKey {
    friend class MetadataCatalog;
    ConstructionKey() = default;
  };

public:
  /// Admit fixed control/payload slabs and the actual shared control block.
  /// max_revision_bytes bounds each complete document, not one incremental delta.
  /// \throws CollectionError for invalid geometry, admission or allocation.
  static std::shared_ptr<MetadataCatalog> create(std::shared_ptr<MemoryBudget> budget, BatchSource source,
                                                 std::size_t capacity = 64, std::size_t max_revision_bytes = 65536);
  /// Retire only after all collection/Writer users and borrowed views end.
  ~MetadataCatalog();
  MetadataCatalog(const MetadataCatalog &) = delete;
  MetadataCatalog &operator=(const MetadataCatalog &) = delete;
  /// Append a caller-validated complete document. Serializes cold registration
  /// only; collection/Writer readers use acquire publication and never lock.
  /// \throws CollectionError for exhausted capacity, size, missing newline,
  /// nonincreasing revision or admission closure; prior revisions stay unchanged.
  MetadataRevision append(std::uint64_t revision, std::span<const std::byte> bytes);
  /// Attach a semantic owner after its exact first document was registered,
  /// before collection begins. Replacement and incomplete owners reject.
  /// \throws CollectionError on invalid registration or closed admission.
  void install_coverage(MetadataCoverage coverage);
  /// Semantic owner's append path. authority must match the installed owner;
  /// ordinary append rejects after attachment, preventing bypass of its
  /// immutable-definition and selected-revision coverage checks.
  /// \throws CollectionError for wrong ownership or normal append failures.
  MetadataRevision append_validated(std::uint64_t revision, std::span<const std::byte> bytes, const void *authority);
  /// Reject subsequent registration. An already admitted append may finish;
  /// original runtime retirement must include that registration caller.
  void stop() noexcept { stopped_.store(true, std::memory_order_release); }
  /// Published prefix length; later appends cannot invalidate earlier views.
  std::size_t size() const noexcept { return count_.load(std::memory_order_acquire); }
  /// Immutable ordinal in that prefix, or no value when not yet published.
  std::optional<MetadataRevision> at(std::size_t index) const noexcept;
  /// Latest complete revision observed after the copy, with its exact hash.
  /// The registry's append-before-new-reference invariant ensures coverage.
  /// \throws CollectionError when no revision exists or source clocks disagree.
  BatchInfo resolve(std::span<const Event> copied) const;
  /// Allocation-free resolver borrowed by a retained collection owner.
  MetadataResolver resolver() noexcept;
  /// Check channel registration against this catalog's exact source and first
  /// published revision/hash before installing either owner into service.
  /// \throws CollectionError with metadata_unavailable on disagreement.
  void require_registration(BatchSource source, const BatchInfo &initial) const;
  /// Cold registration query; immutable after service starts.
  bool has_coverage() const noexcept { return coverage_.owner && coverage_.validate; }
  /// Maximum accepted revision count, fixed before service.
  std::size_t capacity() const noexcept { return capacity_; }
  /// Source provenance, never inferred from the JSON or Event values.
  const BatchSource &source() const noexcept { return source_; }
  /// Allocator-only constructor; use create() for shared control admission.
  MetadataCatalog(ConstructionKey, std::shared_ptr<MemoryBudget> budget, BatchSource source, std::size_t capacity,
                  std::size_t max_revision_bytes);

private:
  friend class CollectionService;
  MetadataRevision append_owned(std::uint64_t revision, std::span<const std::byte> bytes, const void *authority);
  struct Slot {
    std::uint64_t revision = 0;
    std::size_t size = 0;
    std::array<std::byte, 32> sha256{};
  };
  const std::shared_ptr<MemoryBudget> budget_;
  const BatchSource source_;
  const std::size_t capacity_;
  const std::size_t max_revision_bytes_;
  Slot *slots_ = nullptr;
  std::byte *payload_ = nullptr;
  std::atomic<std::size_t> count_{0};
  std::atomic<bool> stopped_{false};
  std::mutex registration_;
  MetadataCoverage coverage_{};
};

} // namespace xpool::devkit::timeline
