#include <xpool/devkit/timeline/service.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <exception>
#include <limits>

#include <sys/mman.h>
#include <unistd.h>

namespace xpool::devkit::timeline {
namespace {
std::atomic<CollectionService *> active{nullptr};
bool same_source(const BatchSource &left, const BatchSource &right) {
  return left.producer == right.producer && left.clock_domain_id == right.clock_domain_id &&
         left.run_id == right.run_id;
}
[[noreturn]] void worker_error(int cause) {
  throw CollectionError(CollectionErrorCode::Allocation, 0, "xpool worker setup failed", std::nullopt,
                        std::strerror(cause));
}
void mark_failure(Collector &collector, CollectionErrorCode code) noexcept {
  if (code == CollectionErrorCode::Transfer)
    collector.fail_transfer();
  else
    collector.fail(code == CollectionErrorCode::Io ? StopReason::IoError : StopReason::MetadataUnavailable);
}
std::chrono::steady_clock::time_point retirement_deadline(std::chrono::steady_clock::time_point production_deadline,
                                                          std::chrono::milliseconds maximum_wait) {
  if (maximum_wait.count() < 0 || maximum_wait.count() > 5000)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0,
                          "xpool shutdown wait requires zero through five seconds");
  return std::min(production_deadline, std::chrono::steady_clock::now() + maximum_wait);
}
} // namespace

CollectionService::Stack::Stack(MemoryBudget &budget) {
  const auto queried = ::sysconf(_SC_PAGESIZE);
  if (queried <= 0)
    worker_error(EINVAL);
  page = static_cast<std::size_t>(queried);
  bytes = 1048576 + page;
  charge = budget.reserve(MemoryKind::Threads, bytes);
  mapping = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    mapping = nullptr;
    worker_error(errno);
  }
  if (::mprotect(mapping, page, PROT_NONE) != 0) {
    const auto cause = errno;
    ::munmap(mapping, bytes);
    mapping = nullptr;
    worker_error(cause);
  }
}
CollectionService::Stack::~Stack() {
  if (mapping)
    ::munmap(mapping, bytes);
}
void CollectionService::Stack::configure(pthread_attr_t &attributes) const {
  const auto result = ::pthread_attr_setstack(&attributes, static_cast<std::byte *>(mapping) + page, bytes - page);
  if (result != 0)
    worker_error(result);
}

std::shared_ptr<CollectionService> CollectionService::create(std::shared_ptr<MemoryBudget> host,
                                                             std::shared_ptr<DiskBudget> disk, std::size_t sources,
                                                             std::chrono::milliseconds cadence,
                                                             std::size_t snapshot_exports) {
  if (!host || !disk || sources == 0 || sources > 1024 || cadence.count() <= 0)
    throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool service geometry is invalid");
  auto result = make_budgeted_shared<CollectionService>(host, MemoryKind::Control, 0, ConstructionKey{}, host, disk,
                                                        sources, cadence, snapshot_exports);
  return result;
}
CollectionService::CollectionService(ConstructionKey, std::shared_ptr<MemoryBudget> host,
                                     std::shared_ptr<DiskBudget> disk, std::size_t sources,
                                     std::chrono::milliseconds cadence, std::size_t snapshot_exports)
    : host_(std::move(host)), disk_(std::move(disk)), capacity_(sources), cadence_(cadence),
      registry_(*host_, MemoryKind::Control), entries_(&registry_), export_capacity_(snapshot_exports),
      snapshot_exports_(&registry_), launches_{{{this, false}, {this, true}}} {
  entries_.resize(capacity_);
  snapshot_exports_.resize(export_capacity_);
  for (auto &stack : stacks_)
    stack = make_budgeted_shared<Stack>(host_, MemoryKind::Control, 0, *host_);
}
CollectionService::~CollectionService() = default;
void CollectionService::require_preparation() const {
  require_registration();
  if (started_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool service cold preparation has closed");
}
void CollectionService::require_registration() const {
  if (stopping_.load(std::memory_order_acquire) || failed())
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool service registration has closed");
}
void CollectionService::add_host(std::shared_ptr<HostChannel> channel, std::shared_ptr<MetadataCatalog> metadata,
                                 std::filesystem::path outdir, std::size_t chunk_bytes,
                                 std::shared_ptr<LegacyRegistry> legacy,
                                 std::span<const std::shared_ptr<SnapshotExport>> exporters) {
  if (!channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool Host registration requires its owner");
  const auto initial = channel->info();
  add({std::move(channel), {}, std::move(metadata), {}, std::move(legacy)}, initial, std::move(outdir), chunk_bytes,
      exporters);
}
void CollectionService::add_device(std::shared_ptr<DeviceChannel> channel, BatchInfo initial,
                                   std::shared_ptr<MetadataCatalog> metadata, std::filesystem::path outdir,
                                   std::size_t chunk_bytes, std::shared_ptr<LegacyRegistry> legacy,
                                   std::span<const std::shared_ptr<SnapshotExport>> exporters) {
  if (!channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool Device registration requires its owner");
  add({{}, std::move(channel), std::move(metadata), {}, std::move(legacy)}, initial, std::move(outdir), chunk_bytes,
      exporters);
}
void CollectionService::add(Entry entry, BatchInfo initial, std::filesystem::path outdir, std::size_t chunk_bytes,
                            std::span<const std::shared_ptr<SnapshotExport>> exporters) {
  const auto lock = std::lock_guard{lifecycle_};
  require_registration();
  const auto &source = entry.collector().source();
  if (!entry.metadata || !entry.metadata->has_coverage())
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool service requires typed semantic metadata coverage");
  if (entry.metadata->budget_ != host_ || (entry.host && entry.host->host_ != host_) ||
      (entry.device && entry.device->host_ != host_))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool every source and metadata owner must share the process Host ledger");
  entry.metadata->require_registration(source, initial);
  const auto count = source_count_.load(std::memory_order_acquire);
  if (count == capacity_)
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "xpool source registry is full");
  for (const auto &previous : std::span{entries_}.first(count))
    if (previous.collector().source().producer == source.producer ||
        previous.collector().source().run_id != source.run_id)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool service source identity conflicts with registration");
  if (entry.device && (entry.device->transfer_uncertain_ || entry.device->retired()))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool Device registration cannot reuse uncertain or retired storage");
  entry.writer = make_budgeted_shared<RawWriter>(host_, MemoryKind::Writer, source.producer, source, std::move(outdir),
                                                 *host_, *disk_, chunk_bytes, entry.metadata->capacity());
  if (entry.legacy) {
    if (entry.legacy->host_ != host_)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool reconstruction requires common Host ledger");
    entry.legacy->require_source(source);
    const auto request_probe = make_budgeted_shared<ReadRequest>(host_, MemoryKind::Snapshot, 0, nullptr, 0, 0, 0);
    entry.legacy->preflight();
  }
  validate_exporters(exporters, source);
  // A running Graph lane cannot observe a half-initialized Device entry. All
  // writer/read owners above coexist before this late Source allocates CUDA.
  if (started_ && entry.device && !entry.device->initialized_)
    entry.device->activate();
  const auto publication = std::lock_guard{wait_mutex_};
  require_registration();
  entries_[count] = std::move(entry);
  // Writer can observe an exporter only after its complete source exists. No
  // worker consults vector end/capacity while lifecycle registration changes.
  source_count_.store(count + 1, std::memory_order_release);
  auto output_count = export_count_.load(std::memory_order_relaxed);
  for (const auto &exporter : exporters)
    snapshot_exports_[output_count++] = {exporter, count};
  export_count_.store(output_count, std::memory_order_release);
  condition_.notify_all();
}
std::optional<CollectionService::DeviceSource> CollectionService::device_source(BatchSource source) {
  const auto lock = std::lock_guard{lifecycle_};
  require_registration();
  for (const auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire))) {
    const auto &registered = entry.collector().source();
    if (registered.run_id != source.run_id ||
        (registered.producer == source.producer && (!same_source(registered, source) || !entry.device)))
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool shared Device Source identity conflicts with registration");
    if (registered.producer == source.producer)
      return DeviceSource{entry.device, entry.metadata, entry.legacy};
  }
  return std::nullopt;
}

void CollectionService::attach_graph_log(std::shared_ptr<GraphLogStream> stream) {
  const auto lock = std::lock_guard{lifecycle_};
  require_preparation();
  if (!stream || graph_log_ || stream->host_ != host_ || stream->disk_ != disk_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool Graph log requires the service's common ledgers and one owner");
  const auto publication = std::lock_guard{wait_mutex_};
  require_registration();
  graph_log_ = std::move(stream);
}
void CollectionService::attach_legacy(std::shared_ptr<LegacyRegistry> registry, BatchSource source) {
  const auto lock = std::lock_guard{lifecycle_};
  require_preparation();
  if (!registry || registry->host_ != host_)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool reconstruction requires common Host ledger");
  registry->require_source(source);
  for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire))) {
    if (entry.collector().source().producer != source.producer)
      continue;
    registry->require_source(entry.collector().source());
    if (entry.legacy)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool source already has reconstruction");
    entry.legacy = std::move(registry);
    return;
  }
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                        "xpool reconstruction source is not registered");
}

void CollectionService::validate_exporters(std::span<const std::shared_ptr<SnapshotExport>> exporters,
                                           BatchSource source) const {
  const auto count = export_count_.load(std::memory_order_acquire);
  if (exporters.size() > export_capacity_ - count)
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool snapshot export registry is full");
  for (auto index = std::size_t{0}; index < exporters.size(); ++index) {
    const auto &exporter = exporters[index];
    if (!exporter || exporter->host_ != host_ || exporter->publisher_->disk_ != disk_ ||
        !same_source(exporter->source(), source))
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                            "xpool snapshot export requires matching source and common ledgers");
    for (const auto &previous : std::span{snapshot_exports_}.first(count))
      if (previous.exporter->path() == exporter->path())
        throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                              "xpool snapshot destination already registered");
    for (const auto &previous : exporters.first(index))
      if (previous->path() == exporter->path())
        throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                              "xpool snapshot destination repeated in registration");
  }
}

void CollectionService::attach_snapshot_export(std::shared_ptr<SnapshotExport> exporter) {
  const auto lock = std::lock_guard{lifecycle_};
  require_registration();
  if (!exporter)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool snapshot export requires an owner");
  const auto source_count = source_count_.load(std::memory_order_acquire);
  for (std::size_t i = 0; i < source_count; ++i) {
    const auto &source = entries_[i].collector().source();
    if (same_source(source, exporter->source())) {
      validate_exporters(std::span{&exporter, 1}, source);
      const auto publication = std::lock_guard{wait_mutex_};
      require_registration();
      const auto count = export_count_.load(std::memory_order_relaxed);
      snapshot_exports_[count] = {std::move(exporter), i};
      export_count_.store(count + 1, std::memory_order_release);
      condition_.notify_all();
      return;
    }
  }
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, exporter->source().producer,
                        "xpool snapshot export source is not registered");
}

std::shared_ptr<SnapshotRead> CollectionService::read_legacy(std::shared_ptr<LegacyView> view, std::uint64_t attempts,
                                                             std::chrono::steady_clock::time_point deadline) {
  return request_read(std::move(view), attempts, deadline);
}

std::shared_ptr<SnapshotExport>
CollectionService::replace_snapshot_export(const std::shared_ptr<SnapshotExport> &previous, detail::LegacyScope scope,
                                           std::size_t capacity,
                                           std::variant<TransportJsonMetadata, FabricJsonMetadata> metadata) {
  const auto lifecycle = std::lock_guard{lifecycle_};
  require_registration();
  const auto count = export_count_.load(std::memory_order_acquire);
  for (auto &entry : std::span{snapshot_exports_}.first(count)) {
    if (entry.exporter != previous)
      continue;
    // Serialize the no-I/O replacement transaction with global stop. The old
    // mailbox mutex rejects a queued/active write before rebinding its publisher.
    const auto publication = std::lock_guard{wait_mutex_};
    require_registration();
    auto replacement = previous->replacement(scope, capacity, metadata);
    {
      const auto exchange = std::lock_guard{export_mutex_};
      entry.exporter = replacement;
    }
    condition_.notify_all();
    return replacement;
  }
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, scope.source.producer,
                        "xpool snapshot replacement requires the currently registered owner");
}

std::shared_ptr<SnapshotRead> CollectionService::read_fabric(std::shared_ptr<LegacyView> view,
                                                             std::chrono::steady_clock::time_point deadline) {
  if (!view || view->scope().family != detail::LegacyFamily::Fabric)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                          "xpool Fabric read requires its registered view");
  return request_read(std::move(view), std::nullopt, deadline);
}

std::uint64_t CollectionService::fabric_attempts(std::size_t source) const {
  const auto &entry = entries_[source];
  const auto progress = entry.collector().fabric_progress();
  if (!entry.device || !progress.exact)
    throw CollectionError(CollectionErrorCode::ResourceBudget, entry.collector().source().producer,
                          "xpool Fabric original operation count is unavailable or exhausted");
  return progress.operations;
}

std::shared_ptr<SnapshotRead> CollectionService::request_read(std::shared_ptr<LegacyView> view,
                                                              std::optional<std::uint64_t> attempts,
                                                              std::chrono::steady_clock::time_point deadline) {
  const auto keep_alive = shared_from_this();
  auto request = std::shared_ptr<ReadRequest>{};
  {
    const auto lifecycle = std::lock_guard{lifecycle_};
    if (!started_ || !view || failed())
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0,
                            "xpool snapshot read requires healthy started service and registered view");
    auto index = std::size_t{0};
    const auto count = source_count_.load(std::memory_order_acquire);
    while (index < count && (!entries_[index].legacy || !entries_[index].legacy->contains(view)))
      ++index;
    if (index == count)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, view->scope().source.producer,
                            "xpool snapshot view is not registered");
    if (joined())
      return view->read_at(attempts ? *attempts : fabric_attempts(index));
    if (stopping_.load(std::memory_order_acquire))
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, view->scope().source.producer,
                            "xpool snapshot read admission closed");
    const auto lock = std::lock_guard{read_mutex_};
    if (read_request_ || next_read_ == std::numeric_limits<std::uint64_t>::max())
      throw CollectionError(CollectionErrorCode::ResourceBudget, view->scope().source.producer,
                            "xpool snapshot read mailbox busy or exhausted");
    request = make_budgeted_shared<ReadRequest>(host_, MemoryKind::Snapshot, view->scope().source.producer, view, index,
                                                attempts, next_read_++);
    read_request_ = request;
  }
  condition_.notify_all();
  auto lock = std::unique_lock{read_mutex_};
  if (!read_condition_.wait_until(lock, deadline, [&] { return request->done; })) {
    request->cancelled.store(true, std::memory_order_release);
    throw CollectionError(CollectionErrorCode::Io, view->scope().source.producer,
                          "xpool snapshot collection cutoff timed out");
  }
  if (request->conflict)
    throw detail::ReconstructionError{*request->conflict};
  if (request->error)
    throw CollectionError(*request->error, view->scope().source.producer,
                          "xpool snapshot cutoff or materialization failed");
  return std::move(request->result);
}

void CollectionService::complete_read(const std::shared_ptr<ReadRequest> &request, bool drained) {
  if (!request)
    return;
  if (!drained && !request->cancelled.load(std::memory_order_acquire) && !failed())
    return;
  auto result = std::shared_ptr<SnapshotRead>{};
  auto error = std::optional<CollectionErrorCode>{};
  auto conflict = std::optional<detail::ReconstructionFailure>{};
  if (!request->cancelled.load(std::memory_order_acquire)) {
    try {
      if (failed())
        rethrow_failure();
      if (entries_[request->source].collector().stop_requested())
        throw CollectionError(CollectionErrorCode::Io, request->view->scope().source.producer,
                              "xpool failed source cannot prove snapshot cutoff");
      result = request->view->read_at(request->attempts ? *request->attempts : fabric_attempts(request->source));
    } catch (const detail::ReconstructionError &cause) {
      conflict = cause.context();
    } catch (const CollectionError &cause) {
      error = cause.code();
    } catch (...) {
      error = CollectionErrorCode::Allocation;
    }
  }
  const auto lock = std::lock_guard{read_mutex_};
  if (!request->cancelled.load(std::memory_order_acquire))
    request->result = std::move(result);
  request->error = error;
  request->conflict = conflict;
  request->done = true;
  if (read_request_ == request)
    read_request_.reset();
  read_condition_.notify_all();
}

void CollectionService::preflight() {
  const auto lock = std::lock_guard{lifecycle_};
  require_registration();
  preflight_unlocked();
}

void CollectionService::preflight_unlocked() {
  // The one pending read control coexists with materialization and handoff.
  const auto request_probe = make_budgeted_shared<ReadRequest>(host_, MemoryKind::Snapshot, 0, nullptr, 0, 0, 0);
  for (const auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire)))
    if (entry.legacy)
      entry.legacy->preflight();
}

void CollectionService::start() {
  const auto lock = std::lock_guard{lifecycle_};
  require_registration();
  if (started_)
    return;
  require_preparation();
  preflight_unlocked();
  for (const auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire)))
    if (entry.device && (!entry.device->initialized_ || entry.device->transfer_uncertain_ || entry.device->retired()))
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, entry.collector().source().producer,
                            "xpool service start requires activated Device sources");
  {
    const auto publication = std::lock_guard{wait_mutex_};
    require_registration();
    auto *expected = static_cast<CollectionService *>(nullptr);
    if (!active.compare_exchange_strong(expected, this))
      throw CollectionError(CollectionErrorCode::ResourceBudget, 0, "xpool process already has collection workers");
    retained_ = shared_from_this();
    started_ = true;
  }
  // Keep the complete owner alive through both actual thread joins.
  for (const auto index : {1U, 0U}) {
    auto attributes = pthread_attr_t{};
    auto result = ::pthread_attr_init(&attributes);
    if (result == 0) {
      try {
        stacks_[index]->configure(attributes);
        result = ::pthread_create(&workers_[index], &attributes, &run, &launches_[index]);
      } catch (...) {
        ::pthread_attr_destroy(&attributes);
        stop();
        failure(CollectionErrorCode::Allocation, 0, "xpool collection worker failed");
        if (!launched_[0] && !launched_[1]) {
          active.store(nullptr, std::memory_order_release);
          retained_.reset();
        }
        throw;
      }
      ::pthread_attr_destroy(&attributes);
    }
    if (result != 0) {
      stop();
      failure(CollectionErrorCode::Allocation, 0, "xpool collection worker creation failed", std::strerror(result));
      if (!launched_[0] && !launched_[1]) {
        active.store(nullptr, std::memory_order_release);
        retained_.reset();
      }
      worker_error(result);
    }
    launched_[index] = true;
  }
}
void CollectionService::stop() noexcept {
  const auto lock = std::lock_guard{wait_mutex_};
  stop_unlocked();
}
void CollectionService::stop_unlocked() noexcept {
  stopping_.store(true, std::memory_order_release);
  if (graph_log_)
    graph_log_->stop();
  for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire))) {
    entry.metadata->stop();
    if (entry.host)
      entry.host->buffer().stop();
    else
      entry.device->stop();
  }
  condition_.notify_all();
}
void CollectionService::failure(const CollectionError &error) noexcept {
  failure(error.code(), error.producer(), error.what_without_backtrace(), error.root_cause());
}
void CollectionService::failure(CollectionErrorCode code, std::uint64_t producer, std::string_view message,
                                std::string_view cause) noexcept {
  if (failure_claimed_.test_and_set(std::memory_order_relaxed))
    return;
  const auto copy = [](auto &destination, std::string_view text) {
    const auto count = std::min(text.size(), destination.size() - 1);
    if (count != 0)
      std::copy_n(text.begin(), count, destination.begin());
    if (text.size() > count)
      std::fill_n(destination.begin() + count - 3, 3, '.');
    destination[count] = '\0';
  };
  failure_producer_ = producer;
  copy(failure_message_, message);
  copy(failure_cause_, cause);
  // One lane owns construction. The release publishes all immutable fields;
  // other lanes never wait for it and readers acquire only the completed code.
  failure_.store(static_cast<std::uint32_t>(code), std::memory_order_release);
}
void CollectionService::rethrow_failure() const {
  const auto code = failure_.load(std::memory_order_acquire);
  if (code != 0)
    throw CollectionError(static_cast<CollectionErrorCode>(code), failure_producer_, failure_message_.data(),
                          std::nullopt, failure_cause_.data());
}
void CollectionService::wait_tick(std::chrono::milliseconds period) {
  auto lock = std::unique_lock{wait_mutex_};
  // Stop publication shares this mutex with entry into the timed wait. A stop
  // that arrives during a copy cannot be lost behind the configured cadence.
  if (stopping_.load(std::memory_order_acquire))
    period = std::min(period, std::chrono::milliseconds{1});
  const auto now = std::chrono::steady_clock::now();
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - now);
  condition_.wait_until(lock, now + std::min(period, remaining));
}
void *CollectionService::run(void *argument) noexcept {
  const auto &launch = *static_cast<Launch *>(argument);
  try {
    if (launch.writer)
      launch.owner->write();
    else
      launch.owner->collect();
  } catch (const CollectionError &error) {
    launch.owner->failure(error);
    launch.owner->worker_failed_.store(true, std::memory_order_release);
    launch.owner->stop();
  } catch (const std::exception &error) {
    launch.owner->failure(CollectionErrorCode::Allocation, 0, "xpool collection worker failed", error.what());
    launch.owner->worker_failed_.store(true, std::memory_order_release);
    launch.owner->stop();
  } catch (...) {
    launch.owner->failure(CollectionErrorCode::Allocation, 0, "xpool collection worker raised an unknown exception");
    launch.owner->worker_failed_.store(true, std::memory_order_release);
    launch.owner->stop();
  }
  return nullptr;
}
void CollectionService::collect() {
  for (;;) {
    auto request = std::shared_ptr<ReadRequest>{};
    {
      const auto lock = std::lock_guard{read_mutex_};
      request = read_request_;
    }
    auto drained = false;
    const auto retired = producers_retired_.load(std::memory_order_acquire);
    auto complete = retired;
    auto in_flight = false;
    for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire))) {
      auto &collector = entry.collector();
      const auto selected = request && &entry == &entries_[request->source];
      if (collector.source_closed() && (!entry.device || entry.device->retired())) {
        drained = drained || selected;
        continue;
      }
      try {
        if (collector.stop_requested()) {
          if (entry.host)
            entry.host->buffer().stop();
          else
            entry.device->stop();
        }
        if (entry.device) {
          if (selected && !request->requested) {
            entry.device->request_drain(request->identity);
            request->requested = true;
          }
          entry.device->tick(retired, entry.metadata->resolver(),
                             entry.legacy ? entry.legacy->sink() : ReconstructionSink{});
          if (selected)
            drained = entry.device->drained(request->identity);
        } else {
          const auto copied =
              collector.collect_host(entry.host->buffer(), entry.host->info(), entry.metadata->resolver(),
                                     entry.legacy ? entry.legacy->sink() : ReconstructionSink{});
          if (selected)
            drained = copied == 0;
          if (retired && copied == 0)
            collector.seal_source(entry.host->buffer().quiescent_counts());
        }
      } catch (const CollectionError &error) {
        failure(error);
        mark_failure(collector, error.code());
      } catch (const std::exception &error) {
        failure(CollectionErrorCode::MetadataUnavailable, collector.source().producer,
                "xpool collection operation failed", error.what());
        collector.fail(StopReason::MetadataUnavailable);
      } catch (...) {
        failure(CollectionErrorCode::MetadataUnavailable, collector.source().producer,
                "xpool collection operation raised an unknown exception");
        collector.fail(StopReason::MetadataUnavailable);
      }
      in_flight = in_flight || (entry.device && entry.device->in_flight());
      complete = complete && collector.source_closed() && (!entry.device || entry.device->retired());
    }
    complete_read(request, drained);
    if (graph_log_) {
      graph_log_->collect();
      if (graph_retired_.load(std::memory_order_acquire) && graph_log_->producer_empty())
        graph_collected_.store(true, std::memory_order_release);
      complete = complete && graph_collected_.load(std::memory_order_acquire);
    }
    if (complete) {
      collection_done_.store(true, std::memory_order_release);
      condition_.notify_all();
      return;
    }
    condition_.notify_all();
    wait_tick(retired || in_flight || request ? std::chrono::milliseconds{1} : cadence_);
  }
}
void CollectionService::write() {
  auto next_quality = std::chrono::steady_clock::now();
  for (;;) {
    const auto collected = collection_done_.load(std::memory_order_acquire);
    auto complete = collected;
    const auto checkpoint = std::chrono::steady_clock::now() >= next_quality;
    const auto outputs = export_count_.load(std::memory_order_acquire);
    for (auto index = std::size_t{0}; index < outputs; ++index) {
      auto entry = ExportEntry{};
      {
        const auto borrow = std::lock_guard{export_mutex_};
        entry = snapshot_exports_[index];
      }
      if (collected)
        entry.exporter->close();
      auto &source = entries_[entry.source];
      try {
        entry.exporter->write_one(source.writer.get(), source.metadata.get());
      } catch (const CollectionError &error) {
        failure(error);
        mark_failure(source.collector(), error.code());
      } catch (const std::exception &error) {
        failure(CollectionErrorCode::Io, entry.exporter->source().producer, "xpool snapshot export failed",
                error.what());
      }
      complete = complete && !entry.exporter->report().pending;
    }
    for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire))) {
      if (entry.finished)
        continue;
      auto &collector = entry.collector();
      try {
        const auto wrote = entry.writer->write_one(collector, entry.metadata.get());
        if (collected && !wrote) {
          if (!collector.stop_requested())
            entry.writer->publish_pending_metadata(*entry.metadata);
          collector.finish_export();
          entry.finished = true;
        } else {
          complete = false;
        }
        if (checkpoint || (collected && !wrote))
          entry.writer->publish_quality(quality(collector));
      } catch (const CollectionError &error) {
        failure(error);
        // RawWriter already attributes data/quota failures and leased losses.
        if (error.code() == CollectionErrorCode::Transfer) {
          collector.fail_transfer();
        } else if (!collector.stop_requested()) {
          // All Writer storage was admitted before start. Runtime budget
          // failures here are disk admission, including quality/metadata files.
          if (error.code() == CollectionErrorCode::ResourceBudget)
            collector.fail(StopReason::DiskQuota);
          else
            mark_failure(collector, error.code());
        }
        complete = false;
      } catch (const std::exception &error) {
        failure(CollectionErrorCode::MetadataUnavailable, collector.source().producer, "xpool writer operation failed",
                error.what());
        collector.fail(StopReason::MetadataUnavailable);
        complete = false;
      } catch (...) {
        failure(CollectionErrorCode::MetadataUnavailable, collector.source().producer,
                "xpool collection operation raised an unknown exception");
        collector.fail(StopReason::MetadataUnavailable);
        complete = false;
      }
    }
    if (graph_log_) {
      const auto graph_collected = graph_collected_.load(std::memory_order_acquire);
      const auto wrote = graph_log_->write_one();
      if (graph_collected && !wrote)
        complete = graph_log_->finish() && complete;
      else
        complete = false;
      if (const auto error = graph_log_->report().error)
        failure(*error, 0, "xpool graph log export failed");
    }
    if (checkpoint)
      next_quality = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    if (complete) {
      condition_.notify_all();
      return;
    }
    wait_tick(cadence_);
  }
}
void CollectionService::begin_retirement(bool producer_retired) {
  // The lifecycle owner serializes registration and both retirement calls.
  stop();
  if (producer_retired) {
    const auto publication = std::lock_guard{wait_mutex_};
    graph_retired_.store(true, std::memory_order_release);
    producers_retired_.store(true, std::memory_order_release);
  }
  condition_.notify_all();
}

bool CollectionService::close_graph(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                                    std::chrono::milliseconds maximum_wait) {
  const auto keep_alive = shared_from_this();
  const auto deadline = retirement_deadline(production_deadline, maximum_wait);
  {
    auto lock = std::unique_lock{lifecycle_, std::defer_lock};
    if (!lock.try_lock_until(deadline))
      return joined();
    if (joined())
      return true;
    if (!graph_log_)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "xpool service has no Graph log");
    const auto publication = std::lock_guard{wait_mutex_};
    graph_log_->stop();
    if (producer_retired)
      graph_retired_.store(true, std::memory_order_release);
    if (source_count_.load(std::memory_order_acquire) == 0) {
      // Freeze registration before extending this Graph-only proof to the
      // whole service. A racing late source must never inherit that proof.
      stop_unlocked();
      if (producer_retired)
        producers_retired_.store(true, std::memory_order_release);
    }
    condition_.notify_all();
    if (!producers_retired_.load(std::memory_order_acquire))
      return false;
  }
  return shutdown(deadline, false, maximum_wait);
}

bool CollectionService::retire_sources(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                                       std::chrono::milliseconds maximum_wait) {
  const auto keep_alive = shared_from_this();
  const auto deadline = retirement_deadline(production_deadline, maximum_wait);
  auto lock = std::unique_lock{lifecycle_, std::defer_lock};
  if (!lock.try_lock_until(deadline))
    return collection_done_.load(std::memory_order_acquire);
  if (collection_done_.load(std::memory_order_acquire))
    return true;
  begin_retirement(producer_retired);
  if (!started_)
    return source_count_.load(std::memory_order_acquire) == 0 && !graph_log_;
  auto waiting = std::unique_lock{wait_mutex_};
  // Collector publication may race entry into this wait. Bound the sleep just
  // as shutdown does; the original deadline is never restarted by a wakeup.
  while (!collection_done_.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire)))
        entry.collector().fail(StopReason::ShutdownDeadline);
      return false;
    }
    condition_.wait_until(waiting, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds{1}));
  }
  return true;
}

bool CollectionService::shutdown(std::chrono::steady_clock::time_point production_deadline, bool producer_retired,
                                 std::chrono::milliseconds maximum_wait) {
  const auto keep_alive = shared_from_this();
  const auto deadline = retirement_deadline(production_deadline, maximum_wait);
  auto lock = std::unique_lock{lifecycle_, std::defer_lock};
  if (!lock.try_lock_until(deadline))
    return joined();
  if (joined())
    return true;
  begin_retirement(producer_retired);
  if (!started_)
    return source_count_.load(std::memory_order_acquire) == 0 && !graph_log_;
  for (;;) {
    auto complete = true;
    for (auto index = std::size_t{0}; index < workers_.size(); ++index) {
      if (launched_[index] && !reaped_[index]) {
        const auto result = ::pthread_tryjoin_np(workers_[index], nullptr);
        if (result == 0)
          reaped_[index] = true;
        else if (result != EBUSY)
          failure(CollectionErrorCode::Allocation, 0, "xpool collection worker join failed", std::strerror(result));
      }
      complete = complete && (!launched_[index] || reaped_[index]);
    }
    if (complete && collection_done_.load(std::memory_order_acquire) &&
        !worker_failed_.load(std::memory_order_acquire)) {
      joined_.store(true, std::memory_order_release);
      active.store(nullptr, std::memory_order_release);
      retained_.reset();
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      for (auto &entry : std::span{entries_}.first(source_count_.load(std::memory_order_acquire)))
        entry.collector().fail(StopReason::ShutdownDeadline);
      return false;
    }
    auto waiting = std::unique_lock{wait_mutex_};
    condition_.wait_until(waiting, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds{1}));
  }
}

} // namespace xpool::devkit::timeline
