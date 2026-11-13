#include <xpool/devkit/timeline/snapshotexport.hpp>

#include <array>
#include <limits>
#include <utility>

namespace xpool::devkit::timeline {

std::shared_ptr<SnapshotExport>
SnapshotExport::create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk, std::string_view path,
                       detail::LegacyScope scope, std::size_t capacity,
                       std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata) {
  return make_budgeted_shared<SnapshotExport>(host, MemoryKind::Writer, scope.source.producer, ConstructionKey{}, host,
                                              std::move(disk), path, scope, capacity, metadata);
}

SnapshotExport::SnapshotExport(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                               std::string_view path, detail::LegacyScope scope, std::size_t capacity,
                               std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata,
                               std::shared_ptr<SnapshotPublisher> publisher)
    : host_(host), scope_(scope), capacity_(capacity),
      formatter_(LegacyJsonFormatter::create(host, scope, capacity, metadata)),
      publisher_(publisher ? std::move(publisher)
                           : SnapshotPublisher::create(std::move(host), std::move(disk), path, scope)) {}

std::shared_ptr<SnapshotExport>
SnapshotExport::replacement(detail::LegacyScope scope, std::size_t capacity,
                            std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata) {
  auto lock = std::unique_lock{mutex_, std::try_to_lock};
  if (!lock.owns_lock() || !report_.closed || report_.pending || replaced_ || publisher_->failed())
    throw CollectionError(CollectionErrorCode::ResourceBudget, scope.source.producer,
                          "xpool snapshot replacement requires a closed, drained mailbox and healthy file owner");
  if (scope.source.run_id != scope_.source.run_id || scope.source.producer != scope_.source.producer ||
      scope.source.clock_domain_id != scope_.source.clock_domain_id || scope.family != scope_.family)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope.source.producer,
                          "xpool snapshot destination cannot change its Source or observer family");
  auto result =
      make_budgeted_shared<SnapshotExport>(host_, MemoryKind::Writer, scope.source.producer, ConstructionKey{}, host_,
                                           nullptr, publisher_->path(), scope, capacity, metadata, publisher_);
  publisher_->rebind(scope);
  replaced_ = true;
  return result;
}

bool SnapshotExport::submit(std::shared_ptr<const SnapshotRead> read) {
  if (!read)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope_.source.producer,
                          "xpool snapshot export requires an immutable read");
  const auto &report = *read->evidence();
  if (!read->belongs_to(scope_) || report.capacity > capacity_ || !report.sequence)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope_.source.producer,
                          "xpool snapshot export read has a different lifecycle or unknown original counts");
  auto lock = std::unique_lock{mutex_, std::try_to_lock};
  if (!lock.owns_lock() || report_.pending || report_.closed || report_.error)
    return false;
  if (report_.accepted == std::numeric_limits<std::uint64_t>::max())
    throw CollectionError(CollectionErrorCode::ResourceBudget, scope_.source.producer,
                          "xpool snapshot export publication counter exhausted");
  pending_ = std::move(read);
  ++report_.accepted;
  report_.pending = true;
  return true;
}

bool SnapshotExport::write_one(RawWriter *metadata_writer, const MetadataCatalog *metadata) {
  if ((metadata_writer == nullptr) != (metadata == nullptr) ||
      (metadata && (metadata->size() == 0 || metadata->source().run_id != scope_.source.run_id ||
                    metadata->source().producer != scope_.source.producer ||
                    metadata->source().clock_domain_id != scope_.source.clock_domain_id)))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope_.source.producer,
                          "xpool snapshot publication requires its complete matching writer/catalog pair");
  auto read = std::shared_ptr<const SnapshotRead>{};
  {
    auto lock = std::lock_guard{mutex_};
    if (!pending_)
      return false;
    read = std::move(pending_);
  }
  auto metadata_ready = metadata_writer == nullptr;
  try {
    if (metadata_writer) {
      metadata_writer->publish_pending_metadata(*metadata);
      metadata_ready = true;
    }
    publisher_->publish(*read, formatter_->document());
  } catch (const CollectionError &error) {
    const auto code = !metadata_ready && error.code() != CollectionErrorCode::ResourceBudget
                          ? CollectionErrorCode::MetadataUnavailable
                          : error.code();
    read.reset();
    {
      auto lock = std::lock_guard{mutex_};
      report_.error = code;
      report_.closed = true;
      report_.pending = false;
    }
    if (code != error.code())
      throw CollectionError(code, scope_.source.producer, "xpool snapshot metadata prerequisites are unavailable",
                            error.path(), error.what());
    throw;
  } catch (...) {
    read.reset();
    auto lock = std::lock_guard{mutex_};
    report_.error = metadata_ready ? CollectionErrorCode::Io : CollectionErrorCode::MetadataUnavailable;
    report_.closed = true;
    report_.pending = false;
    throw;
  }
  // Release only the Writer's reference before reopening its mailbox. Other
  // internal readers keep their actual read/report allocations charged.
  read.reset();
  auto lock = std::lock_guard{mutex_};
  ++report_.published;
  report_.pending = false;
  return true;
}

void SnapshotExport::close() {
  auto lock = std::lock_guard{mutex_};
  report_.closed = true;
}

bool SnapshotExport::close_unavailable(CollectionErrorCode error) {
  auto lock = std::unique_lock{mutex_, std::try_to_lock};
  if (!lock.owns_lock() || report_.pending || replaced_)
    return false;
  if (!report_.error)
    report_.error = error;
  report_.closed = true;
  return true;
}

bool SnapshotExport::belongs_to(const detail::LegacyScope &scope) const noexcept {
  return scope.source.run_id == scope_.source.run_id && scope.source.producer == scope_.source.producer &&
         scope.source.clock_domain_id == scope_.source.clock_domain_id && scope.family == scope_.family &&
         scope.generation == scope_.generation && scope.endpoint_owner == scope_.endpoint_owner &&
         scope.endpoint_sequence == scope_.endpoint_sequence && scope.scheduler == scope_.scheduler;
}

SnapshotExportReport SnapshotExport::report() const {
  auto lock = std::lock_guard{mutex_};
  return report_;
}

} // namespace xpool::devkit::timeline
