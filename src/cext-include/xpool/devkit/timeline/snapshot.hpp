#pragma once

/// \file xpool/devkit/timeline/snapshot.hpp
/// \brief Immutable same-cutoff legacy Records and their versioned evidence report.

#include <memory>
#include <memory_resource>
#include <span>
#include <vector>

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/timeline/evidence.hpp>
#include <xpool/devkit/transport_observer.hpp>

namespace xpool::devkit::timeline {

/// One Host-owned read image. The original runtime's quiescence boundary and
/// the serialized reconstruction lane must both be held throughout create().
/// Afterwards readers retain independent immutable Records and evidence from
/// exactly that cutoff; subsequent merges cannot change this image. Actual
/// Record capacity, report slab and both shared control blocks are charged to
/// the process Host budget until their respective last readers retire.
class SnapshotRead {
  class ConstructionKey {
    friend class SnapshotRead;
    ConstructionKey() = default;
  };

public:
  /// Materialize eligible Records and their evidence in one stable read.
  /// Missing required facts omit only that Record; its report entry remains.
  /// Unknown original attempt totals remain null in the report and cannot be
  /// used to fabricate a complete legacy JSON envelope.
  /// \throws CollectionError on admission/allocation or invalid evidence;
  /// structured retained conflicts propagate before a successful image exists.
  static std::shared_ptr<SnapshotRead> create(std::shared_ptr<MemoryBudget> budget, const detail::LegacyStore &store,
                                              SnapshotMetadata metadata = {});
  SnapshotRead(const SnapshotRead &) = delete;
  SnapshotRead &operator=(const SnapshotRead &) = delete;
  /// Only the selected observer has Records; the other span is empty.
  std::span<const xpool::devkit::transport_observer::Record> transport_records() const noexcept { return transport_; }
  /// Immutable role-specific Fabric values, sorted by shared local trace ID.
  std::span<const xpool::devkit::fabric_observer::Record> fabric_records() const noexcept { return fabric_; }
  /// Same-cutoff report. A reader can retain it independently of the Records.
  const std::shared_ptr<const SnapshotEvidenceReport> &evidence() const noexcept { return evidence_; }
  /// Match the exact registered Run/Producer/clock and observer lifetime.
  bool belongs_to(const detail::LegacyScope &scope) const noexcept;
  /// Actual reserved Record storage, excluding separately charged report and
  /// shared control blocks. Useful for complete startup/read peak admission.
  std::size_t record_bytes() const noexcept;
  /// Allocator-only constructor; use create() for budgeted control ownership.
  SnapshotRead(ConstructionKey, std::shared_ptr<MemoryBudget> budget, const detail::LegacyStore &store,
               SnapshotMetadata metadata);

private:
  const std::shared_ptr<MemoryBudget> budget_;
  BudgetMemoryResource resource_;
  const std::shared_ptr<const SnapshotEvidenceReport> evidence_;
  std::pmr::vector<xpool::devkit::transport_observer::Record> transport_;
  std::pmr::vector<xpool::devkit::fabric_observer::Record> fabric_;
};

} // namespace xpool::devkit::timeline
