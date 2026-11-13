#include <xpool/devkit/timeline/legacyview.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace xpool::devkit::timeline {

namespace {

std::optional<detail::ReconstructionFailure> validate_events(std::span<const Event> events, BatchSource source) {
  for (const auto &event : events) {
    try {
      event.validate(source.producer, source.clock_domain_id);
    } catch (const ParseError &error) {
      return detail::ReconstructionFailure{
          error.code() == ParseErrorCode::UnknownKind ? detail::ReconstructionErrorCode::UnknownKind
                                                      : detail::ReconstructionErrorCode::InvalidEvent,
          source.producer, event.event_sequence, std::nullopt, "invalid source Event structure"};
    }
  }
  return std::nullopt;
}

bool matches(const detail::LegacyScope &scope, const Event &event) {
  constexpr auto generation = std::uint32_t{1} << 0;
  constexpr auto endpoint = std::uint32_t{1} << 1;
  constexpr auto invocation = std::uint32_t{1} << 5;
  if ((event.presence & generation) != 0 && event.generation != scope.generation)
    return false;
  if (scope.family == detail::LegacyFamily::Fabric)
    return (event.presence & (generation | invocation)) == (generation | invocation);
  const auto role = scope.family == detail::LegacyFamily::TransportInstance ? 1U : 2U;
  return event.role == role && (event.presence & endpoint) != 0 && event.endpoint_owner == scope.endpoint_owner &&
         event.endpoint_sequence == scope.endpoint_sequence;
}

bool same_domain(const detail::LegacyScope &left, const detail::LegacyScope &right) {
  return left.family == right.family &&
         (left.family == detail::LegacyFamily::Fabric
              ? left.generation == right.generation
              : left.endpoint_owner == right.endpoint_owner && left.endpoint_sequence == right.endpoint_sequence);
}

} // namespace

std::shared_ptr<LegacyView> LegacyView::create(std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                                               std::size_t capacity, SnapshotMetadata metadata) {
  return make_budgeted_shared<LegacyView>(host, MemoryKind::Snapshot, scope.source.producer, ConstructionKey{}, host,
                                          scope, capacity, metadata);
}

LegacyView::LegacyView(ConstructionKey, std::shared_ptr<MemoryBudget> host, detail::LegacyScope scope,
                       std::size_t capacity, SnapshotMetadata metadata)
    : host_(std::move(host)), store_(host_, scope, capacity), metadata_(metadata) {
  preflight();
}

void LegacyView::preflight() const {
  const auto lock = std::lock_guard{mutex_};
  const auto read = SnapshotRead::create(host_, store_, metadata_);
}

void LegacyView::merge(std::span<const Event> events) {
  if (const auto failure = validate_events(events, scope().source)) {
    fail(*failure);
    throw detail::ReconstructionError{*failure};
  }
  merge_validated(events);
}

void LegacyView::fail(detail::ReconstructionFailure failure) {
  const auto lock = std::lock_guard{mutex_};
  if (!failure_)
    failure_ = failure;
}

void LegacyView::merge_validated(std::span<const Event> events) {
  const auto lock = std::lock_guard{mutex_};
  if (failure_)
    throw detail::ReconstructionError{*failure_};
  for (const auto &event : events) {
    if (matches(store_.scope(), event)) {
      // A temporary quiescent read does not permanently retire production.
      // A later original operation invalidates that cutoff's total until the
      // next independent counter copy; it is not a counter-regression error.
      if (event.local_trace_id > store_.counts().sequence)
        store_.forget_attempts();
      if (event.role == 3 && event.kind >= 300 && event.kind <= 307 && scope().scheduler) {
        const auto ticket = (event.presence & (std::uint32_t{1} << 13)) != 0;
        const auto fifo = scope().scheduler == detail::LegacyPolicy::Fifo;
        if (ticket != fifo || (fifo && event.ready_ticket == 0)) {
          failure_ = detail::ReconstructionFailure{detail::ReconstructionErrorCode::InvalidEvent,
                                                   scope().source.producer, event.event_sequence, std::nullopt,
                                                   "Coordinator ticket disagrees with its immutable scheduler policy"};
          throw detail::ReconstructionError{*failure_};
        }
      }
      store_.merge(event);
    }
  }
}

void LegacyView::observe_attempts(std::uint64_t total) {
  const auto lock = std::lock_guard{mutex_};
  store_.observe_attempts(total);
}

void LegacyView::forget_attempts() {
  const auto lock = std::lock_guard{mutex_};
  store_.forget_attempts();
}

std::shared_ptr<SnapshotRead> LegacyView::read() {
  const auto lock = std::lock_guard{mutex_};
  if (failure_)
    throw detail::ReconstructionError{*failure_};
  return SnapshotRead::create(host_, store_, metadata_);
}

std::shared_ptr<SnapshotRead> LegacyView::read_at(std::uint64_t attempts) {
  const auto lock = std::lock_guard{mutex_};
  if (failure_)
    throw detail::ReconstructionError{*failure_};
  store_.observe_attempts(attempts);
  return SnapshotRead::create(host_, store_, metadata_);
}

LegacyResourceReport LegacyView::resources() const {
  const auto lock = std::lock_guard{mutex_};
  return {store_.reconstruction_bytes(), store_.provenance_bytes()};
}

bool LegacyRegistry::contains(const std::shared_ptr<LegacyView> &view) {
  const auto lock = std::lock_guard{mutex_};
  return std::ranges::find(views_, view) != views_.end();
}

void LegacyRegistry::preflight() {
  const auto lock = std::lock_guard{mutex_};
  // All automatic-export mailboxes can retain their immutable reads at once.
  // Probe their coexistence, including the temporary handle array itself.
  auto reads = std::pmr::vector<std::shared_ptr<SnapshotRead>>{&resource_};
  reads.reserve(views_.size());
  for (const auto &view : views_)
    reads.push_back(view->read());
}

std::shared_ptr<LegacyRegistry> LegacyRegistry::create(std::shared_ptr<MemoryBudget> host, BatchSource source,
                                                       std::size_t capacity) {
  return make_budgeted_shared<LegacyRegistry>(host, MemoryKind::Snapshot, source.producer, ConstructionKey{}, host,
                                              source, capacity);
}

LegacyRegistry::LegacyRegistry(ConstructionKey, std::shared_ptr<MemoryBudget> host, BatchSource source,
                               std::size_t capacity)
    : host_(std::move(host)), source_(source), capacity_(capacity),
      resource_(*host_, MemoryKind::Snapshot, source.producer), views_(&resource_) {
  if (source.producer == 0 || source.clock_domain_id == 0 || capacity == 0)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool legacy registry needs a source and fixed capacity");
  if (capacity > views_.max_size())
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool legacy registry capacity overflows");
  views_.reserve(capacity);
}

void LegacyRegistry::require_source(const BatchSource &source) const {
  if (source.producer != source_.producer || source.run_id != source_.run_id ||
      source.clock_domain_id != source_.clock_domain_id)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool legacy registry belongs to another source");
}

std::shared_ptr<LegacyView> LegacyRegistry::add(detail::LegacyScope scope, std::size_t capacity,
                                                SnapshotMetadata metadata) {
  require_source(scope.source);
  const auto lock = std::lock_guard{mutex_};
  if (failure_)
    throw detail::ReconstructionError{*failure_};
  if (views_.size() == capacity_ ||
      std::ranges::any_of(views_, [&](const auto &view) { return same_domain(view->scope(), scope); }))
    throw CollectionError(CollectionErrorCode::ResourceBudget, source_.producer,
                          "xpool legacy view registration exceeds capacity or repeats a domain");
  auto view = LegacyView::create(host_, scope, capacity, metadata);
  views_.push_back(view);
  return view;
}

void LegacyRegistry::retire(const std::shared_ptr<LegacyView> &view) {
  const auto lock = std::lock_guard{mutex_};
  const auto found = std::ranges::find(views_, view);
  if (found == views_.end())
    throw CollectionError(CollectionErrorCode::InvalidRelease, source_.producer,
                          "xpool legacy view retirement has no matching registry owner");
  views_.erase(found);
}

ReconstructionSink LegacyRegistry::sink() noexcept {
  return {this,
          [](void *context, std::span<const Event> events) { static_cast<LegacyRegistry *>(context)->merge(events); }};
}

void LegacyRegistry::merge(std::span<const Event> events) {
  const auto lock = std::lock_guard{mutex_};
  if (failure_)
    throw detail::ReconstructionError{*failure_};
  if (views_.empty())
    return;
  if (const auto failure = validate_events(events, source_)) {
    failure_ = failure;
    for (const auto &view : views_)
      view->fail(*failure);
    throw detail::ReconstructionError{*failure};
  }
  try {
    for (const auto &view : views_)
      view->merge_validated(events);
  } catch (const detail::ReconstructionError &error) {
    failure_ = error.context();
    for (const auto &view : views_)
      view->fail(*failure_);
    throw;
  }
}

} // namespace xpool::devkit::timeline
