#pragma once

/// \file xpool/devkit/timeline/graphsession.hpp
/// \brief Native lifetime of an independent Python Host Graph compatibility log.

#include <memory>
#include <string_view>

#include <xpool/devkit/timeline/options.hpp>
#include <xpool/devkit/timeline/service.hpp>

namespace xpool::devkit::timeline {

/// Own the bounded compatibility pipeline used by the SGLang Python adapter.
/// It shares the process service with raw collection, performs no CUDA
/// initialization, and declares no C0 Producer or unified Graph capability.
/// Graph admission can close while raw Sources continue collecting.
class GraphLogSession {
  class ConstructionKey {
    friend class GraphLogSession;
    ConstructionKey() = default;
  };

public:
  /// Prepare the Graph log and start the shared service before raw installation.
  /// path is an absolute owned JSONL destination. Numeric limits come from
  /// xpool.config's frozen timeline policy;
  /// unified enablement is rejected until the production providers are installed.
  /// Sessions share fixed process Host/disk ledgers, including caller-retained
  /// closed sessions. A destination is admitted once by its preparation owner;
  /// Python repeat install reuses its active session. Existing destination files
  /// remain untouched and are charged during setup.
  /// \throws CollectionError for invalid policy, admission or worker setup.
  static std::shared_ptr<GraphLogSession> create(std::string_view path, const Options &options);
  /// Bounded no-I/O enqueue; phase/kind and backend limits match GraphLogStream.
  bool record(std::uint32_t phase, std::uint32_t kind, std::string_view backend, std::uint64_t time_ns) noexcept {
    return stream_->record(phase, kind, backend, time_ns);
  }
  /// Close admission before establishing actual producer-call retirement.
  void stop() noexcept { stream_->stop(); }
  /// Close within the first process observation deadline (at most five seconds).
  /// The caller supplies true only after all enqueue calls have retired. Timeout preserves
  /// the live service and its process claim; close can be retried later.
  /// A mixed service stays live until its raw producers have separately retired;
  /// closing this log supplies no Device proof. True means both worker joins,
  /// independently of report().error/lost. Repeated calls never renew the wait.
  bool close(bool producers_retired);
  /// Independent compatibility counters, never C0 raw Event quality.
  GraphLogReport report() const noexcept { return stream_->report(); }
  /// Allocator-only constructor; call create().
  GraphLogSession(ConstructionKey, std::shared_ptr<CollectionService> service, std::shared_ptr<GraphLogStream> stream,
                  std::chrono::milliseconds wait);

private:
  const std::shared_ptr<CollectionService> service_;
  const std::shared_ptr<GraphLogStream> stream_;
  const std::chrono::milliseconds wait_;
};
} // namespace xpool::devkit::timeline
