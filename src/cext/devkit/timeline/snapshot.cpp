#include <xpool/devkit/timeline/snapshot.hpp>

#include <utility>

#include <xpool/devkit/timeline/legacyaccess.hpp>

namespace xpool::devkit::timeline {

std::shared_ptr<SnapshotRead> SnapshotRead::create(std::shared_ptr<MemoryBudget> budget,
                                                   const detail::LegacyStore &store, SnapshotMetadata metadata) {
  return make_budgeted_shared<SnapshotRead>(budget, MemoryKind::Snapshot, store.scope().source.producer,
                                            ConstructionKey{}, budget, store, metadata);
}

SnapshotRead::SnapshotRead(ConstructionKey, std::shared_ptr<MemoryBudget> budget, const detail::LegacyStore &store,
                           SnapshotMetadata metadata)
    : budget_(std::move(budget)), resource_(*budget_, MemoryKind::Snapshot, store.scope().source.producer),
      evidence_(SnapshotEvidenceReport::create(budget_, store, metadata)), transport_(&resource_), fabric_(&resource_) {
  const auto fabric = evidence_->observer == SnapshotObserver::Fabric;
  if (fabric)
    fabric_.reserve(store.capacity());
  else
    transport_.reserve(store.capacity());
  for (const auto &entry : evidence_->entries()) {
    if (entry.projection_status != ProjectionStatus::Retained)
      continue;
    const auto operation = *store.read(entry.local_trace_id);
    if (fabric)
      fabric_.push_back(
          *detail::LegacyRecordAccess::project<xpool::devkit::fabric_observer::Record>(operation, store.scope()));
    else
      transport_.push_back(
          *detail::LegacyRecordAccess::project<xpool::devkit::transport_observer::Record>(operation, store.scope()));
  }
}

std::size_t SnapshotRead::record_bytes() const noexcept {
  return transport_.capacity() * sizeof(xpool::devkit::transport_observer::Record) +
         fabric_.capacity() * sizeof(xpool::devkit::fabric_observer::Record);
}

bool SnapshotRead::belongs_to(const detail::LegacyScope &scope) const noexcept {
  const auto &report = *evidence_;
  const auto fabric = scope.family == detail::LegacyFamily::Fabric;
  const auto site =
      scope.family == detail::LegacyFamily::TransportInstance ? SnapshotRole::Instance : SnapshotRole::AtnAgent;
  return report.source.producer == scope.source.producer && report.source.run_id == scope.source.run_id &&
         report.source.clock_domain_id == scope.source.clock_domain_id && report.generation == scope.generation &&
         (report.observer == SnapshotObserver::Fabric) == fabric &&
         (fabric ||
          (report.site == site && report.endpoint == std::array{scope.endpoint_owner, scope.endpoint_sequence}));
}

} // namespace xpool::devkit::timeline
