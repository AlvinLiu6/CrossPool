#pragma once

/// \file xpool/devkit/timeline/legacyview.hpp
/// \brief Bounded live reconstruction with budgeted immutable reads.

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <xpool/devkit/timeline/sink.hpp>
#include <xpool/devkit/timeline/snapshot.hpp>

namespace xpool::devkit::timeline {

/// View-owned reconstruction storage and the checked transient read peak.
struct LegacyResourceReport {
  /// Retained fixed reconstruction table bytes, excluding provenance.
  std::size_t reconstruction_bytes;
  /// Retained raw reference/provenance storage bytes.
  std::size_t provenance_bytes;
};

/// One registered Endpoint/site or Fabric Producer/Generation view. A short
/// Host mutex serializes completed-copy merging and same-cutoff read creation.
/// Original runtime quiescence remains a caller precondition for read/counts;
/// the mutex alone cannot prove production or CUDA retirement.
class LegacyView {
  class ConstructionKey {
    friend class LegacyView;
    ConstructionKey() = default;
  };

public:
  /// Admit reconstruction, provenance, shared control and a complete read
  /// peak before service. Preflight actual Native read/control allocations.
  /// \throws CollectionError on invalid registration or Host admission.
  static std::shared_ptr<LegacyView> create(std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                                            std::size_t capacity, SnapshotMetadata metadata = {});
  LegacyView(const LegacyView &) = delete;
  LegacyView &operator=(const LegacyView &) = delete;
  /// Merge matching local facts from one completed source-bound copy. Unrelated
  /// Endpoint/Generation/family values do not enter this view or its ID window.
  /// Conflicts retain bounded context and fail later reads explicitly.
  void merge(std::span<const Event> events);
  /// Supply the original record-begin count at its production quiescence
  /// boundary, independently of Event loss. Serialize with collection merges.
  /// \throws CollectionError for regression or a count below an observed ID.
  void observe_attempts(std::uint64_t total);
  /// Invalidate a previous cutoff's exact original total without discarding its
  /// monotonic lower bound or altering already materialized immutable reads.
  void forget_attempts();
  /// Create an immutable read while the original source is quiescent and the
  /// Collector has completed a fresh drain. Every live read remains charged;
  /// subsequent reads are admitted against the common remaining Host budget.
  std::shared_ptr<SnapshotRead> read();
  /// Atomically install original attempts and freeze the same completed cutoff.
  std::shared_ptr<SnapshotRead> read_at(std::uint64_t attempts);
  /// Check the maximum internal read peak before starting service.
  void preflight() const;
  /// Fixed domain retained through Endpoint/Generation recreation.
  const detail::LegacyScope &scope() const noexcept { return store_.scope(); }
  /// Consistent itemization while completed copies or readers progress.
  LegacyResourceReport resources() const;
  /// Allocator-only constructor; use create() for startup admission.
  LegacyView(ConstructionKey, std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope, std::size_t capacity,
             SnapshotMetadata metadata);

private:
  friend class LegacyRegistry;
  void merge_validated(std::span<const Event> events);
  void fail(detail::ReconstructionFailure failure);
  const std::shared_ptr<MemoryBudget> host_;
  detail::LegacyStore store_;
  const SnapshotMetadata metadata_;
  friend class CollectionService;
  std::optional<detail::ReconstructionFailure> failure_ = std::nullopt;
  mutable std::mutex mutex_;
};

/// Source-bound bounded dispatch to registered views. Registration/retirement
/// serialize with collection; no new registry slots or buffers grow at runtime.
class LegacyRegistry {
  class ConstructionKey {
    friend class LegacyRegistry;
    ConstructionKey() = default;
  };

public:
  /// Preallocate view handles before collection starts.
  /// \throws CollectionError for invalid source/capacity or admission failure.
  static std::shared_ptr<LegacyRegistry> create(std::shared_ptr<MemoryBudget> host, BatchSource source,
                                                std::size_t capacity = 8);
  LegacyRegistry(const LegacyRegistry &) = delete;
  LegacyRegistry &operator=(const LegacyRegistry &) = delete;
  /// Register one unique source/domain, including its checked read peak,
  /// before the original runtime emits related Events.
  /// \throws CollectionError on duplicate/wrong source, capacity or admission.
  std::shared_ptr<LegacyView> add(detail::LegacyScope scope, std::size_t capacity, SnapshotMetadata metadata = {});
  /// Remove a view only after the owning runtime proves its last relevant
  /// source references and collection copies retired. Existing read/view owners
  /// retain their own storage and budgets after removal.
  /// \throws CollectionError when this registry does not own that exact view.
  void retire(const std::shared_ptr<LegacyView> &view);
  /// Borrow a collection callback; its channel keeps this registry alive.
  ReconstructionSink sink() noexcept;
  /// Validate the channel's exact registered Run/Producer/clock.
  /// \throws CollectionError with metadata_unavailable on another source.
  void require_source(const BatchSource &source) const;
  /// Allocator-only constructor; use create() for complete control admission.
  LegacyRegistry(ConstructionKey, std::shared_ptr<MemoryBudget> host, BatchSource source, std::size_t capacity);

private:
  friend class CollectionService;
  bool contains(const std::shared_ptr<LegacyView> &view);
  void preflight();
  void merge(std::span<const Event> events);
  const std::shared_ptr<MemoryBudget> host_;
  const BatchSource source_;
  const std::size_t capacity_;
  BudgetMemoryResource resource_;
  std::pmr::vector<std::shared_ptr<LegacyView>> views_;
  std::optional<detail::ReconstructionFailure> failure_ = std::nullopt;
  std::mutex mutex_;
};

} // namespace xpool::devkit::timeline
