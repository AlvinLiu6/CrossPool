#include <xpool/devkit/timeline/graphsession.hpp>

#include <utility>

#include <unistd.h>

#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {
std::shared_ptr<GraphLogSession> GraphLogSession::create(std::string_view path, const Options &options) {
  auto &preparation = ProcessPreparation::singleton();
  const auto lock = std::lock_guard{preparation.mutex_};
  preparation.require_graph_policy(options);
  if (preparation.policy_)
    preparation.acquire(options.host_bytes, options.disk_bytes, options.metadata_reserve_bytes);
  auto host = preparation.host_ ? preparation.host_ : MemoryBudget::create(options.host_bytes);
  auto disk = preparation.disk_ ? preparation.disk_
                                : make_budgeted_shared<DiskBudget>(host, MemoryKind::Control, 0, options.disk_bytes,
                                                                   options.metadata_reserve_bytes);
  auto resident =
      preparation.host_ ? MemoryBudget::Reservation{} : host->reserve(MemoryKind::Control, sizeof(preparation));
  auto stream = GraphLogStream::create(host, disk, path, 0, static_cast<std::uint64_t>(::getpid()));
  const auto endpoints =
      preparation.policy_ ? preparation.policy_->limits.endpoint_capacity : options.endpoint_capacity;
  auto service =
      preparation.acquire_service(host, disk, std::chrono::milliseconds{options.collection_interval_ms},
                                  xpool::utils::checked::sum(static_cast<std::size_t>(endpoints), std::size_t{1}));
  service->attach_graph_log(stream);
  auto session = make_budgeted_shared<GraphLogSession>(host, MemoryKind::Control, 0, ConstructionKey{}, service, stream,
                                                       std::chrono::milliseconds{options.shutdown_wait_ms});
  // Commit first-use ledgers only after Graph setup succeeds, preserving retry
  // after failed standalone preparation with no installed bootstrap policy.
  if (!preparation.host_) {
    preparation.host_ = host;
    preparation.disk_ = disk;
    preparation.resident_ = std::move(resident);
  }
  service->start();
  return session;
}
GraphLogSession::GraphLogSession(ConstructionKey, std::shared_ptr<CollectionService> service,
                                 std::shared_ptr<GraphLogStream> stream, std::chrono::milliseconds wait)
    : service_(std::move(service)), stream_(std::move(stream)), wait_(wait) {}
bool GraphLogSession::close(bool producers_retired) {
  stop();
  auto &preparation = ProcessPreparation::singleton();
  const auto deadline = preparation.begin_retirement(wait_);
  if (!service_->close_graph(deadline, producers_retired, wait_))
    return false;
  const auto lock = std::lock_guard{preparation.mutex_};
  // The expired weak reference would retain the budgeted shared allocation
  // after the last session releases the joined service.
  if (preparation.service_.lock() == service_)
    preparation.service_.reset();
  return true;
}
} // namespace xpool::devkit::timeline
