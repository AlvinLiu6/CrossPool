#pragma once

/// \file xpool/devkit/timeline/legacyjson.hpp
/// \brief Budgeted streaming adapters for the existing legacy snapshot JSON.

#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <xpool/devkit/timeline/snapshotwriter.hpp>

namespace xpool::devkit::timeline {

/// Stable Transport envelope facts copied at lifecycle registration.
struct TransportJsonMetadata {
  /// Positive operating-system process identity used by the legacy document.
  std::uint64_t pid;
  /// Canonical model identifier, copied rather than borrowed during service.
  std::string_view model_id;
  /// Instance rank represented by this endpoint.
  std::uint64_t rank;
  /// Last sixteen characters of the canonical arena handle.
  std::string_view arena_handle_suffix;
};

/// Stable Fabric envelope facts copied at Generation registration.
struct FabricJsonMetadata {
  /// Existing legacy Generation UUID spelling: thirty-two lowercase hex digits.
  std::string_view generation;
  /// NVSHMEM PE, distinct from the registered Producer identity.
  std::uint64_t pe;
  /// Joined attention-side participant count.
  std::uint64_t atnagent_count;
  /// Joined FFN-side participant count.
  std::uint64_t ffnagent_count;
  /// Existing model topology order, copied into charged fixed storage.
  std::span<const xpool::devkit::fabric_observer::ModelTopology> model_topologies;
};

/// Writer-lane formatter bound to one registered lifecycle. Copies envelope
/// metadata and preallocates one capacity-sized duration scratch slab before
/// service. Each serialization streams one immutable read without retaining a
/// JSON document or allocating per Record. Calls are serialized by the Writer.
class LegacyJsonFormatter {
  class ConstructionKey {
    friend class LegacyJsonFormatter;
    ConstructionKey() = default;
  };

public:
  /// Admit all control, metadata and percentile scratch before construction.
  /// The selected metadata alternative must match the scope family. Caller
  /// supplies actual registered facts; no process/Generation identity is inferred.
  /// \throws CollectionError for missing facts, mismatched domain or budget.
  static std::shared_ptr<LegacyJsonFormatter> create(std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                                                     std::size_t capacity,
                                                     std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata);
  LegacyJsonFormatter(const LegacyJsonFormatter &) = delete;
  LegacyJsonFormatter &operator=(const LegacyJsonFormatter &) = delete;
  /// Stream the unchanged legacy JSON shape with a final newline. Raw zero
  /// timestamps remain zero and do not yield a legacy duration. Unknown original
  /// counters or a different lifecycle reject before writing output.
  /// \throws CollectionError for mismatched identity, missing counts or capacity.
  std::size_t serialize(const SnapshotRead &read, JsonSink sink = {});
  /// Borrow this formatter for synchronous two-pass publication. This owner
  /// must remain alive until SnapshotPublisher::publish returns.
  SnapshotDocument document() noexcept;
  /// Allocator-only constructor; use create() to admit shared ownership.
  LegacyJsonFormatter(ConstructionKey, std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                      std::size_t capacity, std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata);

private:
  const std::shared_ptr<MemoryBudget> host_;
  const detail::LegacyScope scope_;
  const std::size_t capacity_;
  BudgetMemoryResource resource_;
  std::pmr::string identity_;
  std::pmr::string suffix_;
  std::uint64_t pid_or_pe_ = 0;
  std::uint64_t rank_or_atnagent_count_ = 0;
  std::uint64_t ffnagent_count_ = 0;
  std::pmr::vector<xpool::devkit::fabric_observer::ModelTopology> topologies_;
  std::pmr::vector<std::uint64_t> durations_;
};

} // namespace xpool::devkit::timeline
