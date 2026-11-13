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

#include <xpool/devkit/timeline/devicechannel.hpp>
#include <xpool/devkit/timeline/graphlog.hpp>
#include <xpool/devkit/timeline/hostchannel.hpp>
#include <xpool/devkit/timeline/legacyview.hpp>
#include <xpool/devkit/timeline/snapshotexport.hpp>
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
                                                   std::chrono::milliseconds cadence = std::chrono::milliseconds{100},
                                                   std::size_t snapshot_exports = 16);
  ~CollectionService();
  CollectionService(const CollectionService &) = delete;
  CollectionService &operator=(const CollectionService &) = delete;
  /// Admit and publish a complete Host source, including optional reconstruction
  /// and exporters. Before start, attach_legacy/export may complete the entry.
  /// After start, the supplied entry is immutable immediately on success. The
  /// first revision/hash must agree; byte-only catalogs reject. Failed admission
  /// leaves the registry unchanged. Stop forbids subsequent publication.
  void add_host(std::shared_ptr<HostChannel> channel, std::shared_ptr<MetadataCatalog> metadata,
                std::filesystem::path outdir, std::size_t chunk_bytes, std::shared_ptr<LegacyRegistry> legacy = {},
                std::span<const std::shared_ptr<SnapshotExport>> exporters = {});
  /// Register a prepared Device channel under the same rules. A running service
  /// activates a cold channel only after Writer/read/export admission, before
  /// publishing it to workers. Production binding/lifetime remain runtime-owned.
  /// An unstarted service defers activation to its runtime before start().
  void add_device(std::shared_ptr<DeviceChannel> channel, BatchInfo initial, std::shared_ptr<MetadataCatalog> metadata,
                  std::filesystem::path outdir, std::size_t chunk_bytes, std::shared_ptr<LegacyRegistry> legacy = {},
                  std::span<const std::shared_ptr<SnapshotExport>> exporters = {});
  /// Borrow the existing Device Source when another observer joins the same
  /// process. These are the registered owners, not a second pool or collector.
  /// The caller still owns actual production quiescence and lifetime proofs.
  struct DeviceSource {
    /// Shared event pool, transfer state and Source retirement owner.
    std::shared_ptr<DeviceChannel> channel;
    /// Immutable revisions already bound to this Source's registered metadata.
    std::shared_ptr<MetadataCatalog> metadata;
    /// Common reconstruction registry to which an observer adds its own view.
    std::shared_ptr<LegacyRegistry> legacy;
  };
  /// Return the exact registered Device Source or null when none exists. A
  /// conflicting Producer/Run/clock or stopped registration rejects explicitly.
  std::optional<DeviceSource> device_source(BatchSource source);
  /// Attach the process compatibility log before workers start. It uses the
  /// same Collector/Writer pair and must share both exact process ledgers.
  void attach_graph_log(std::shared_ptr<GraphLogStream> stream);
  /// Attach bounded reconstruction before start. It shares the process Host
  /// ledger and exact registered source. Late sources provide it through add().
  void attach_legacy(std::shared_ptr<LegacyRegistry> registry, BatchSource source);
  /// Publish one bounded compatibility exporter for an already registered
  /// source, before or after start. It must share the ledgers and use a unique
  /// admitted output path; exporter capacity is fixed by create().
  void attach_snapshot_export(std::shared_ptr<SnapshotExport> exporter);
  /// Reuse one registered destination for a new Endpoint lifecycle. The old
  /// mailbox must be closed and have no queued or Writer-held read. Retain its
  /// publisher and disk charges; admit the new formatter while old owners still
  /// count against Host memory. Replacement uses the same fixed service slot.
  /// Failure leaves the old registration intact. Stop forbids replacement.
  std::shared_ptr<SnapshotExport>
  replace_snapshot_export(const std::shared_ptr<SnapshotExport> &previous, detail::LegacyScope scope,
                          std::size_t capacity, std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata);
  /// Caller holds original source quiescence through return and supplies its
  /// independent record-begin count. A fresh Collector drain precedes freezing
  /// Records/report together. One pending request; timeout cancels delivery,
  /// retaining worker-owned state until actual completion. No file I/O wait.
  std::shared_ptr<SnapshotRead> read_legacy(std::shared_ptr<LegacyView> view, std::uint64_t attempts,
                                            std::chrono::steady_clock::time_point deadline);
  /// Same cutoff protocol, with the Fabric original count taken from that
  /// completed D2H frame. Caller holds actual production quiescence through
  /// return; no independent synchronous CUDA counter copy is performed.
  std::shared_ptr<SnapshotRead> read_fabric(std::shared_ptr<LegacyView> view,
                                            std::chrono::steady_clock::time_point deadline);
  /// Claim the sole active process service and start the two lanes once.
  /// An already running healthy service is reused without another thread pair.
  /// A partial launch failure retains all started workers and their owners.
  void start();
  /// Check complete registered read/control/export-read coexistence before a
  /// new lifecycle starts producing, including late Endpoint registration.
  /// Starts no workers or CUDA; start() repeats this against current occupancy.
  void preflight();
  /// Close source/registration admission; this establishes no producer retirement.
  void stop() noexcept;
  /// Stop only Graph enqueue admission and publish only its producer proof.
  /// With raw sources still active this returns false and leaves their admission,
  /// Device ownership and service threads intact. True still means both worker
  /// joins, not merely a finished Graph file. Standalone Graph can close the
  /// whole service; a mixed service needs the runtime's all-source proof first.
  bool close_graph(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                   std::chrono::milliseconds maximum_wait = std::chrono::milliseconds{5000});
  /// Close admission and wait only for final collection: every Source's last
  /// empty scan, reconstruction merge and Device storage retirement. The
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
                    std::size_t sources, std::chrono::milliseconds cadence, std::size_t snapshot_exports);

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
    std::shared_ptr<LegacyRegistry> legacy;
    bool finished = false;
    Collector &collector() const noexcept { return host ? host->collector() : device->collector(); }
  };
  struct ExportEntry {
    std::shared_ptr<SnapshotExport> exporter;
    std::size_t source;
  };
  struct ReadRequest {
    std::shared_ptr<LegacyView> view;
    std::size_t source;
    std::optional<std::uint64_t> attempts;
    std::uint64_t identity;
    bool requested = false; // Collector lane only.
    std::atomic<bool> cancelled{false};
    bool done = false; // Remaining fields protected by read_mutex_.
    std::optional<CollectionErrorCode> error;
    std::optional<detail::ReconstructionFailure> conflict;
    std::shared_ptr<SnapshotRead> result;
    ReadRequest(std::shared_ptr<LegacyView> selected, std::size_t index, std::optional<std::uint64_t> total,
                std::uint64_t id)
        : view(std::move(selected)), source(index), attempts(total), identity(id) {}
  };
  void complete_read(const std::shared_ptr<ReadRequest> &request, bool drained);
  std::shared_ptr<SnapshotRead> request_read(std::shared_ptr<LegacyView> view, std::optional<std::uint64_t> attempts,
                                             std::chrono::steady_clock::time_point deadline);
  std::uint64_t fabric_attempts(std::size_t source) const;
  struct Launch {
    CollectionService *owner;
    bool writer;
  };
  void add(Entry entry, BatchInfo initial, std::filesystem::path outdir, std::size_t chunk_bytes,
           std::span<const std::shared_ptr<SnapshotExport>> exporters);
  void require_registration() const;
  void validate_exporters(std::span<const std::shared_ptr<SnapshotExport>> exporters, BatchSource source) const;
  void require_preparation() const;
  void preflight_unlocked();
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
  std::shared_ptr<GraphLogStream> graph_log_;
  const std::size_t export_capacity_;
  std::pmr::vector<ExportEntry> snapshot_exports_;
  // Only the registered handle is exchanged under this lock. The Writer owns
  // its borrowed handle through I/O, without holding a lifecycle/table lock.
  std::mutex export_mutex_;
  std::atomic<std::size_t> export_count_{0};
  std::atomic<bool> graph_retired_{false};
  std::atomic<bool> graph_collected_{false};
  std::mutex read_mutex_;
  std::condition_variable read_condition_;
  std::shared_ptr<ReadRequest> read_request_;
  std::uint64_t next_read_ = 1;
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
