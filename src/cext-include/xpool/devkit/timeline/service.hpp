#pragma once

/// \file xpool/devkit/timeline/service.hpp
/// \brief One bounded process Collector/Writer pair with explicit retirement.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

#include <pthread.h>

#include <xpool/devkit/timeline/artifact.hpp>
#include <xpool/devkit/timeline/devicechannel.hpp>
#include <xpool/devkit/timeline/hostchannel.hpp>
#include <xpool/devkit/timeline/writer.hpp>

namespace xpool::devkit::timeline {

/// Prepared channels and immutable metadata share one process service. Source
/// registrations publish complete entries in fixed storage; its two lanes never
/// hold a file-I/O lock in common. Runtime owners prove producer retirement.
class CollectionService : public std::enable_shared_from_this<CollectionService> {
  class ConstructionKey {
    friend class CollectionService;
    ConstructionKey() = default;
  };

public:
  /// Prepare fixed registry and both explicitly budgeted worker stacks. No
  /// workers, files or CUDA initialization occur here.
  static std::shared_ptr<CollectionService> create(std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                                                   std::size_t sources = 8,
                                                   std::chrono::milliseconds cadence = std::chrono::milliseconds{100});
  ~CollectionService();
  CollectionService(const CollectionService &) = delete;
  CollectionService &operator=(const CollectionService &) = delete;
  /// Admit and publish a complete Host source and its immutable metadata.
  /// After start, the supplied entry is immutable immediately on success. The
  /// first revision/hash must agree; byte-only catalogs reject. Failed admission
  /// leaves the registry unchanged. Stop forbids subsequent publication.
  void add_host(std::shared_ptr<HostChannel> channel, std::shared_ptr<MetadataCatalog> metadata,
                std::filesystem::path outdir, std::size_t chunk_bytes);
  /// Register a prepared Device channel under the same rules. A running service
  /// activates a cold channel only after Writer admission, before
  /// publishing it to workers. Production binding/lifetime remain runtime-owned.
  /// An unstarted service defers activation to its runtime before start().
  void add_device(std::shared_ptr<DeviceChannel> channel, BatchInfo initial, std::shared_ptr<MetadataCatalog> metadata,
                  std::filesystem::path outdir, std::size_t chunk_bytes);
  /// Borrow the existing Device Source when another observer joins the same
  /// process. These are the registered owners, not a second pool or collector.
  /// The caller still owns actual production quiescence and lifetime proofs.
  struct DeviceSource {
    /// Shared event pool, transfer state and Source retirement owner.
    std::shared_ptr<DeviceChannel> channel;
    /// Immutable revisions already bound to this Source's registered metadata.
    std::shared_ptr<MetadataCatalog> metadata;
  };
  /// Return the exact registered Device Source or null when none exists. A
  /// conflicting Producer/Run/clock or stopped registration rejects explicitly.
  std::optional<DeviceSource> device_source(BatchSource source);
  /// Bind one enabled evidence class to its independent collection directory.
  /// Uses the existing Source/Writer and shared ledgers, including late joins.
  void register_directory(BatchSource source, EvidenceClass kind, const std::filesystem::path &outdir);
  /// Bounded attachment handoff; full/closed admission rejects explicitly.
  void submit_artifact(std::shared_ptr<ArtifactJob> job);
  /// Claim the sole active process service and start the two lanes once.
  /// An already running healthy service is reused without another thread pair.
  /// A partial launch failure retains all started workers and their owners.
  void start();
  /// Close source/registration admission; this establishes no producer retirement.
  void stop() noexcept;
  /// Close admission and wait only for final collection: every Source's last
  /// empty scan and Device storage retirement. The
  /// producer_retired proof and deadline have the same meaning as shutdown().
  /// True permits dependent runtime retirement; it proves neither publication
  /// nor worker joins. Writer-held Host data, budgets and the process service
  /// claim stay owned until shutdown() and actual owner destruction. Timeout
  /// retains all unfinished owners. A prepared service with registered resources
  /// cannot complete before its workers start; this call never starts them.
  bool retire_sources(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                      std::chrono::milliseconds maximum_wait = std::chrono::milliseconds{5000});
  /// Wait no later than min(production_deadline, now+maximum_wait), capped at
  /// five seconds. producer_retired is an external proof covering every source
  /// API caller, kernel, Graph reference and registration call. False never
  /// seals source totals. Timeout retains the process claim, workers, channels,
  /// buffers, CUDA handles and budget charges. A later call can finish joining.
  /// True means actual resource retirement; failed() still reports export loss.
  /// An unstarted service succeeds only when it has no registered resources.
  bool shutdown(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                std::chrono::milliseconds maximum_wait = std::chrono::milliseconds{5000});
  /// Any typed worker/publication failure, independent of successful joining.
  bool failed() const noexcept { return failure_.load(std::memory_order_acquire) != 0; }
  /// Rethrow the first published worker failure through CollectionError, or
  /// return if none is published. Retain its code, Producer, operation and
  /// cause in 512 fixed text bytes within this budgeted owner; overlong text
  /// ends in "...". Filesystem paths and exception backtraces are not retained.
  /// Concurrent callers read the immutable record after failed() becomes true.
  void rethrow_failure() const;
  /// Last resources are no longer owned by worker threads only after real joins.
  bool joined() const noexcept { return joined_.load(std::memory_order_acquire); }
  /// Actual common ledgers, for preparation of further cold components.
  const std::shared_ptr<MemoryBudget> &host_budget() const noexcept { return host_; }
  /// Common artifact quota retained by every attached Writer/export owner.
  const std::shared_ptr<DiskBudget> &disk_budget() const noexcept { return disk_; }
  /// Allocator-only constructor; call create().
  CollectionService(ConstructionKey, std::shared_ptr<MemoryBudget> host, std::shared_ptr<DiskBudget> disk,
                    std::size_t sources, std::chrono::milliseconds cadence);

private:
  friend class ProcessPreparation;
  struct Stack {
    explicit Stack(MemoryBudget &budget);
    ~Stack();
    void configure(pthread_attr_t &attributes) const;
    MemoryBudget::Reservation charge;
    void *mapping = nullptr;
    std::size_t bytes = 0;
    std::size_t page = 0;
  };
  struct Entry {
    std::shared_ptr<HostChannel> host;
    std::shared_ptr<DeviceChannel> device;
    std::shared_ptr<MetadataCatalog> metadata;
    std::shared_ptr<RawWriter> writer;
    bool finished = false;
    Collector &collector() const noexcept { return host ? host->collector() : device->collector(); }
  };
  struct Launch {
    CollectionService *owner;
    bool writer;
  };
  void add(Entry entry, BatchInfo initial, std::filesystem::path outdir, std::size_t chunk_bytes);
  void require_registration() const;
  void require_preparation() const;
  void stop_unlocked() noexcept;
  void begin_retirement(bool producer_retired);
  void collect();
  void write();
  void wait_tick(std::chrono::milliseconds period);
  void failure(const CollectionError &error) noexcept;
  void failure(CollectionErrorCode code, std::uint64_t producer, std::string_view message,
               std::string_view cause = {}) noexcept;
  static void *run(void *argument) noexcept;
  const std::shared_ptr<MemoryBudget> host_;
  const std::shared_ptr<DiskBudget> disk_;
  const std::size_t capacity_;
  const std::chrono::milliseconds cadence_;
  BudgetMemoryResource registry_;
  std::pmr::vector<Entry> entries_;
  // Storage never resizes. Lifecycle calls construct an unused entry, then
  // publish its complete prefix under wait_mutex_, shared with stop().
  std::atomic<std::size_t> source_count_{0};
  std::mutex artifact_mutex_;
  std::array<std::shared_ptr<ArtifactJob>, 64> artifacts_{};
  std::size_t artifact_head_ = 0;
  std::size_t artifact_count_ = 0;
  std::array<std::shared_ptr<Stack>, 2> stacks_;
  std::array<pthread_t, 2> workers_{};
  std::array<bool, 2> launched_{};
  std::array<bool, 2> reaped_{};
  std::array<Launch, 2> launches_;
  std::timed_mutex lifecycle_;
  std::mutex wait_mutex_;
  std::condition_variable condition_;
  std::shared_ptr<CollectionService> retained_;
  bool started_ = false;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> producers_retired_{false};
  std::atomic<bool> collection_done_{false};
  std::atomic<bool> joined_{false};
  std::atomic<std::uint32_t> failure_{0};
  std::atomic_flag failure_claimed_ = ATOMIC_FLAG_INIT;
  std::uint64_t failure_producer_ = 0;
  std::array<char, 192> failure_message_{};
  std::array<char, 320> failure_cause_{};
  std::atomic<bool> worker_failed_{false};
};

} // namespace xpool::devkit::timeline
