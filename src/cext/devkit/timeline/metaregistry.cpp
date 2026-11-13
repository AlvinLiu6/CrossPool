#include <algorithm>
#include <charconv>
#include <cstdio>
#include <limits>

#include <xpool/devkit/timeline/digest.hpp>
#include <xpool/devkit/timeline/metastate.hpp>

namespace xpool::devkit::timeline {
namespace {
bool run_relative_path(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.find('\0') != path.npos)
    return false;
  auto depth = std::size_t{0};
  while (!path.empty()) {
    const auto end = path.find('/');
    const auto part = path.substr(0, end);
    if (part == "..") {
      if (depth == 0)
        return false;
      --depth;
    } else if (!part.empty() && part != ".") {
      ++depth;
    }
    if (end == path.npos)
      break;
    path.remove_prefix(end + 1);
  }
  return depth != 0;
}

bool same_source(const BatchSource &left, const BatchSource &right) {
  return left.run_id == right.run_id && left.producer == right.producer &&
         left.clock_domain_id == right.clock_domain_id;
}
bool same_process(const RegisteredProcess &left, const RegisteredProcess &right) {
  return left.process == right.process && same_source(left.host, right.host);
}

} // namespace

detail::MetadataState::MetadataState(std::shared_ptr<MemoryBudget> budget, BatchSource registration,
                                     CollectionPolicy configured, MetadataCapabilities available,
                                     MetadataLimits geometry)
    : host(std::move(budget)), source(registration), policy(std::move(configured)), capabilities(available),
      limits(geometry), resource(*host, MemoryKind::Metadata, source.producer), producers(&resource), clocks(&resource),
      generations(&resource), endpoints(&resource), templates(&resource), artifacts(&resource) {
  producers.reserve(limits.producers);
  clocks.reserve(limits.clocks);
  generations.reserve(limits.generations);
  endpoints.reserve(limits.endpoints);
  templates.reserve(limits.templates);
  artifacts.reserve(limits.artifacts);
  document = static_cast<std::byte *>(
      host->allocate_storage(limits.document_bytes, 256, MemoryKind::Metadata, source.producer));
}

detail::MetadataState::~MetadataState() {
  host->deallocate_storage(document, limits.document_bytes, 256, MemoryKind::Metadata);
}

[[noreturn]] void detail::MetadataState::invalid(std::string_view message) const {
  throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer, std::string{message});
}
[[noreturn]] void detail::MetadataState::invalid_membership(std::string_view code, std::uint64_t generation,
                                                            std::optional<std::int32_t> pe, std::uint64_t producer,
                                                            std::string_view detail) const {
  auto message = std::string{code} + " at generations[id=" + std::to_string(generation) + "].pe_members";
  if (pe)
    message += "[pe=" + std::to_string(*pe) + "]";
  if (producer != 0)
    message += " producer=" + std::to_string(producer);
  message += ": ";
  message += detail;
  invalid(message);
}
void detail::MetadataState::capacity(bool available) const {
  if (!available)
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool typed metadata capacity exhausted");
}
const detail::MetadataProducer *detail::MetadataState::producer(std::uint64_t id) const {
  for (const auto &value : producers)
    if (value.source.producer == id)
      return &value;
  return nullptr;
}
const detail::MetadataClock *detail::MetadataState::clock(std::uint64_t id) const {
  for (const auto &value : clocks)
    if (value.id == id)
      return &value;
  return nullptr;
}
const detail::MetadataGeneration *detail::MetadataState::generation(std::uint64_t id) const {
  for (const auto &value : generations)
    if (value.id == id)
      return &value;
  return nullptr;
}
const detail::MetadataEndpoint *detail::MetadataState::endpoint(std::uint64_t owner, std::uint64_t sequence) const {
  for (const auto &value : endpoints)
    if (value.owner == owner && value.sequence == sequence)
      return &value;
  return nullptr;
}
const detail::MetadataTemplate *detail::MetadataState::graph_template(std::uint64_t owner,
                                                                      std::uint64_t sequence) const {
  for (const auto &value : templates)
    if (value.owner == owner && value.sequence == sequence)
      return &value;
  return nullptr;
}
const detail::MetadataArtifact *detail::MetadataState::artifact(std::uint64_t id) const {
  for (const auto &value : artifacts)
    if (value.id == id)
      return &value;
  return nullptr;
}

void detail::MetadataState::add_clock(std::uint64_t id, bool device, IdentityUuid owner, std::uint32_t logical_device,
                                      std::uint64_t epoch, std::string_view physical) {
  if (id == 0 || (device && (epoch == 0 || physical.empty())))
    invalid("xpool metadata clock has invalid identity or Device facts");
  if (const auto *previous = clock(id)) {
    if (previous->device != device || previous->owner != owner || previous->logical_device != logical_device ||
        previous->epoch != epoch || previous->physical != physical)
      invalid("xpool metadata clock definition cannot change");
    return;
  }
  capacity(clocks.size() < limits.clocks);
  clocks.push_back({id, device, owner, logical_device, epoch, std::pmr::string{physical, &resource}});
  changed();
}

void detail::MetadataState::add_host(RegisteredProcess process) {
  if (process.process.pid == 0 || !xpool::is_valid(process.process.runtime_role) ||
      process.process.runtime_role == xpool::RuntimeRole::Daemon || process.host.run_id != source.run_id ||
      process.host.producer == 0 || process.host.clock_domain_id == 0)
    invalid("xpool metadata Host Producer has unsupported process role or invalid registered facts");
  if (const auto *previous = producer(process.host.producer)) {
    if (previous->device || !same_source(previous->source, process.host) || !same_process(previous->process, process))
      invalid("xpool metadata Producer definition cannot change");
    return;
  }
  for (const auto &previous : producers)
    if (previous.process.process.creation_id == process.process.creation_id && !same_process(previous.process, process))
      invalid("xpool metadata process creation has conflicting Host Source facts");
  capacity(producers.size() < limits.producers);
  add_clock(process.host.clock_domain_id, false, process.process.host_creation_id, 0, 0, {});
  producers.push_back({process, process.host, false});
  changed();
}

std::shared_ptr<MetadataRegistry> MetadataRegistry::create(std::shared_ptr<MemoryBudget> host, BatchSource source,
                                                           CollectionPolicy policy, MetadataCapabilities capabilities,
                                                           MetadataLimits limits) {
  const auto sizes =
      std::array{limits.producers,     limits.clocks,    limits.generations,       limits.models_per_generation,
                 limits.endpoints,     limits.templates, limits.compute_templates, limits.artifacts,
                 limits.document_bytes};
  if (source.producer == 0 || source.clock_domain_id == 0)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "xpool typed metadata requires a registered Source");
  constexpr auto maximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!host || limits.producers > maximum / sizeof(detail::MetadataProducer) ||
      limits.clocks > maximum / sizeof(detail::MetadataClock) ||
      limits.generations > maximum / sizeof(detail::MetadataGeneration) ||
      limits.models_per_generation > maximum / sizeof(detail::MetadataModel) ||
      limits.endpoints > maximum / sizeof(detail::MetadataEndpoint) ||
      limits.templates > maximum / sizeof(detail::MetadataTemplate) ||
      limits.compute_templates > maximum / sizeof(std::uint64_t) ||
      limits.artifacts > maximum / sizeof(detail::MetadataArtifact) || limits.document_bytes > maximum ||
      std::ranges::any_of(sizes, [](auto value) { return value == 0; }))
    throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer,
                          "xpool typed metadata geometry/source is invalid");
  try {
    auto state = make_budgeted_shared<detail::MetadataState>(host, MemoryKind::Metadata, source.producer, host, source,
                                                             std::move(policy), capabilities, limits);
    return make_budgeted_shared<MetadataRegistry>(host, MemoryKind::Metadata, source.producer, ConstructionKey{},
                                                  std::move(state));
  } catch (const CollectionError &) {
    throw;
  } catch (const std::exception &error) {
    throw CollectionError(CollectionErrorCode::Allocation, source.producer, "xpool typed metadata allocation failed",
                          std::nullopt, error.what());
  }
}
MetadataRegistry::MetadataRegistry(ConstructionKey, std::shared_ptr<detail::MetadataState> state)
    : state_(std::move(state)) {}
const BatchSource &MetadataRegistry::source() const noexcept { return state_->source; }

void MetadataRegistry::validate_preparation(std::optional<int> device_index) const {
  const auto lock = std::lock_guard{state_->mutex};
  const auto *producer = state_->producer(state_->source.producer);
  const auto *clock = state_->clock(state_->source.clock_domain_id);
  if (state_->catalog_bound || !producer || !clock || producer->device != device_index.has_value() ||
      clock->device != device_index.has_value() ||
      (device_index && (*device_index < 0 || clock->logical_device != static_cast<std::uint32_t>(*device_index))))
    state_->invalid("xpool Source preparation requires an unbound registry with its actual channel kind/mapping");
}

void MetadataRegistry::add_host(RegisteredProcess process) {
  const auto lock = std::lock_guard{state_->mutex};
  state_->add_host(process);
}
void MetadataRegistry::add_device(RegisteredProcess process, BatchSource source, std::uint32_t logical_device,
                                  std::string_view physical_device_uuid, std::uint64_t raw_epoch) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  if (source.run_id != state.source.run_id || source.producer == 0 || source.clock_domain_id == 0 ||
      source.producer == process.host.producer || physical_device_uuid.empty() || raw_epoch == 0)
    state.invalid("xpool metadata Device Producer has invalid registered facts");
  state.add_host(process);
  if (const auto *previous = state.producer(source.producer)) {
    const auto *clock = state.clock(source.clock_domain_id);
    if (!previous->device || !same_source(previous->source, source) || !same_process(previous->process, process) ||
        !clock || clock->owner != process.process.creation_id || clock->logical_device != logical_device ||
        clock->epoch != raw_epoch || clock->physical != physical_device_uuid)
      state.invalid("xpool metadata Device Producer definition cannot change");
    return;
  }
  state.capacity(state.producers.size() < state.limits.producers);
  for (const auto &previous : state.producers)
    if (previous.device && previous.process.process.creation_id == process.process.creation_id) {
      const auto *clock = state.clock(previous.source.clock_domain_id);
      if (clock && clock->logical_device == logical_device)
        state.invalid("xpool process/Device already has a different registered Producer");
    }
  state.add_clock(source.clock_domain_id, true, process.process.creation_id, logical_device, raw_epoch,
                  physical_device_uuid);
  state.producers.push_back({process, source, true});
  state.changed();
}

void MetadataRegistry::add_generation(std::uint64_t id, std::string_view production, std::uint64_t atn_count,
                                      std::uint64_t ffn_count, bool fifo) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  if (id == 0 || production.size() != 32 ||
      production.find_first_not_of("0123456789abcdef") != std::string_view::npos || atn_count == 0 || ffn_count == 0)
    state.invalid("xpool metadata Generation requires exact production identity and positive counts");
  if (const auto *previous = state.generation(id)) {
    if (previous->production != production || previous->atn_count != atn_count || previous->ffn_count != ffn_count ||
        previous->fifo != fifo)
      state.invalid("xpool metadata Generation definition cannot change");
    return;
  }
  state.capacity(state.generations.size() < state.limits.generations);
  auto entry = detail::MetadataGeneration{id,
                                          std::pmr::string{production, &state.resource},
                                          atn_count,
                                          ffn_count,
                                          fifo,
                                          std::pmr::vector<detail::MetadataModel>{&state.resource},
                                          std::pmr::vector<MetadataPeMember>{&state.resource}};
  entry.models.reserve(state.limits.models_per_generation);
  entry.pe_members.reserve(state.limits.producers / 2);
  state.generations.push_back(std::move(entry));
  state.changed();
}
void MetadataRegistry::add_pe_member(std::uint64_t generation, std::int32_t pe, std::uint64_t host_producer,
                                     std::uint64_t device_producer) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  auto found = std::ranges::find(state.generations, generation, &detail::MetadataGeneration::id);
  if (found == state.generations.end() || pe < 0 || host_producer == 0 || device_producer == 0)
    state.invalid_membership("invalid_event", generation, pe, host_producer,
                             "requires a Generation, nonnegative PE and positive Producers");
  const auto member = MetadataPeMember{pe, host_producer, device_producer};
  for (const auto &previous : found->pe_members) {
    if (previous == member)
      return;
    if (previous.pe == pe || previous.host_producer == host_producer || previous.device_producer == device_producer ||
        previous.host_producer == device_producer || previous.device_producer == host_producer)
      state.invalid_membership("identity_conflict", generation, pe, host_producer, "duplicate PE or Producer binding");
  }
  if (host_producer == device_producer)
    state.invalid_membership("identity_conflict", generation, pe, host_producer, "repeated Producer in one member");
  if (found->since != 0)
    state.invalid_membership("identity_conflict", generation, pe, host_producer,
                             "published Generation table cannot change");
  state.capacity(found->pe_members.size() < state.limits.producers / 2);
  const auto position = std::ranges::lower_bound(found->pe_members, pe, {}, &MetadataPeMember::pe);
  found->pe_members.insert(position, member);
  state.changed();
}

std::optional<std::int32_t> MetadataRegistry::fabric_pe(std::uint64_t generation, std::uint64_t producer) const {
  const auto lock = std::lock_guard{state_->mutex};
  state_->validate();
  const auto *definition = state_->generation(generation);
  if (!definition || !state_->producer(producer))
    state_->invalid("xpool Fabric PE lookup requires registered Generation and Producer");
  for (const auto &member : definition->pe_members)
    if (member.host_producer == producer || member.device_producer == producer)
      return member.pe;
  return std::nullopt;
}

void MetadataRegistry::require_fabric(std::uint64_t generation, const xpool::fabric::ArenaProjection &projection,
                                      int pe) const {
  const auto lock = std::lock_guard{state_->mutex};
  state_->validate();
  const auto *definition = state_->generation(generation);
  const auto *producer = state_->producer(state_->source.producer);
  auto name = std::array<char, 33>{};
  std::snprintf(name.data(), name.size(), "%016llx%016llx", static_cast<unsigned long long>(projection.generation_high),
                static_cast<unsigned long long>(projection.generation_low));
  if (!definition || !producer || !producer->device || state_->capabilities.fabric != MetadataCapability::Enabled ||
      definition->production != name.data() || definition->atn_count != projection.atnagent_count ||
      definition->ffn_count != projection.ffnagent_count || definition->models.size() != projection.instances.size() ||
      definition->fifo != (projection.scheduler.type() == xpool::fabric::SchedulerPolicy::Fifo))
    state_->invalid("xpool Fabric recording metadata differs from the actual join projection");
  const auto member = std::ranges::find(definition->pe_members, pe, &MetadataPeMember::pe);
  if (member == definition->pe_members.end() || member->device_producer != state_->source.producer)
    state_->invalid("xpool Fabric recording Source differs from the actual PE");
  for (auto index = std::size_t{0}; index < projection.instances.size(); ++index) {
    const auto &model = definition->models[index];
    const auto &instance = projection.instances[index];
    if (model.instance_index != index || model.tp != instance.atn_tp_size || model.dp != instance.atn_dp_size)
      state_->invalid("xpool Fabric recording model topology differs from the actual join projection");
    if (model.ffn_tp && std::ranges::any_of(instance.layers, [&model](const auto &layer) {
          return layer.ffnagent_indices.size() != *model.ffn_tp;
        }))
      state_->invalid("xpool Fabric recording FFN topology differs from the actual join projection");
  }
}

void MetadataRegistry::add_model(std::uint64_t generation, std::uint64_t index, std::string_view model_id,
                                 std::uint64_t tp, std::uint64_t dp, std::optional<std::uint64_t> ffn_tp) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  auto found = std::ranges::find(state.generations, generation, &detail::MetadataGeneration::id);
  if (found == state.generations.end() || model_id.empty() || tp == 0 || dp == 0 || (ffn_tp && *ffn_tp == 0))
    state.invalid("xpool metadata model/topology requires an existing Generation and positive sizes");
  if (index < found->models.size()) {
    const auto &previous = found->models[static_cast<std::size_t>(index)];
    if (previous.model_id != model_id || previous.tp != tp || previous.dp != dp || previous.ffn_tp != ffn_tp)
      state.invalid("xpool metadata model/topology definition cannot change");
    return;
  }
  if (found->since != 0 || index != found->models.size())
    state.invalid("xpool metadata model list is fixed at Generation publication and must be contiguous");
  state.capacity(found->models.size() < state.limits.models_per_generation);
  found->models.push_back({index, std::pmr::string{model_id, &state.resource}, tp, dp, ffn_tp});
  state.changed();
}

void MetadataRegistry::add_endpoint(std::uint64_t owner, std::uint64_t sequence, std::uint64_t generation,
                                    std::uint64_t index, std::uint32_t rank, std::string_view model_id,
                                    std::string_view suffix, std::uint64_t instance_producer,
                                    std::optional<std::uint64_t> supersedes) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  if (owner == 0 || sequence == 0 || generation == 0 || model_id.empty() || instance_producer == 0 ||
      suffix.size() != 16 || suffix.find_first_not_of("0123456789abcdef") != std::string_view::npos ||
      (supersedes && (*supersedes == 0 || *supersedes == sequence)))
    state.invalid("xpool metadata Endpoint has invalid identity, model or Arena suffix");
  if (const auto *previous = state.endpoint(owner, sequence)) {
    if (previous->generation != generation || previous->instance_index != index || previous->rank != rank ||
        previous->model_id != model_id ||
        std::string_view{previous->suffix.data(), previous->suffix.size()} != suffix ||
        previous->instance_producer != instance_producer || previous->supersedes != supersedes)
      state.invalid("xpool metadata Endpoint definition cannot change");
    return;
  }
  state.capacity(state.endpoints.size() < state.limits.endpoints);
  auto entry = detail::MetadataEndpoint{
      owner, sequence,          generation, index, rank, std::pmr::string{model_id, &state.resource},
      {},    instance_producer, supersedes};
  std::ranges::copy(suffix, entry.suffix.begin());
  state.endpoints.push_back(std::move(entry));
  state.changed();
}

void MetadataRegistry::add_template(std::uint64_t owner, std::uint64_t sequence, std::string_view kind,
                                    std::optional<std::string_view> backend, std::optional<std::uint64_t> signature,
                                    std::optional<std::uint64_t> lane) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  if (owner == 0 || sequence == 0 || kind.empty() || (backend && backend->empty()))
    state.invalid("xpool metadata Graph template has invalid identity or kind");
  if (const auto *previous = state.graph_template(owner, sequence)) {
    if (previous->kind != kind || previous->has_backend != backend.has_value() ||
        previous->backend != backend.value_or("") || previous->signature != signature || previous->lane != lane)
      state.invalid("xpool metadata Graph template definition cannot change");
    return;
  }
  state.capacity(state.templates.size() < state.limits.templates);
  auto entry = detail::MetadataTemplate{owner,
                                        sequence,
                                        std::pmr::string{kind, &state.resource},
                                        std::pmr::string{backend.value_or(""), &state.resource},
                                        backend.has_value(),
                                        signature,
                                        lane,
                                        std::pmr::vector<std::uint64_t>{&state.resource}};
  entry.compute.reserve(state.limits.compute_templates);
  state.templates.push_back(std::move(entry));
  state.changed();
}
void MetadataRegistry::add_compute_template(std::uint64_t owner, std::uint64_t parent, std::uint64_t child) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  auto found = std::ranges::find_if(
      state.templates, [&](const auto &entry) { return entry.owner == owner && entry.sequence == parent; });
  if (found == state.templates.end() || child == 0)
    state.invalid("xpool metadata compute template requires its parent and positive identity");
  if (std::ranges::find(found->compute, child) != found->compute.end())
    return;
  if (found->since != 0)
    state.invalid("xpool published metadata compute-template list cannot change");
  state.capacity(found->compute.size() < state.limits.compute_templates);
  found->compute.push_back(child);
  state.changed();
}
void MetadataRegistry::add_artifact(std::uint64_t id, std::string_view kind, MetadataArtifactStatus status,
                                    std::optional<std::string_view> path,
                                    std::optional<std::array<std::byte, 32>> hash) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  if (id == 0 || kind.empty() || (path && !run_relative_path(*path)) ||
      (status == MetadataArtifactStatus::Published && (!path || !hash)))
    state.invalid("xpool metadata artifact requires valid identity, Run-relative file path and published outcome");
  if (const auto *previous = state.artifact(id)) {
    if (previous->kind != kind || previous->status != status || previous->has_path != path.has_value() ||
        previous->path != path.value_or("") || previous->sha256 != hash)
      state.invalid("xpool metadata artifact definition cannot change");
    return;
  }
  state.capacity(state.artifacts.size() < state.limits.artifacts);
  state.artifacts.push_back({id, std::pmr::string{kind, &state.resource}, status,
                             std::pmr::string{path.value_or(""), &state.resource}, path.has_value(), hash});
  state.changed();
}

std::span<const std::byte> MetadataRegistry::document(std::uint64_t revision) {
  const auto lock = std::lock_guard{state_->mutex};
  return state_->render(revision);
}
void detail::MetadataState::mark_registered(std::uint64_t revision) {
  const auto mark = [&](auto &entries) {
    for (auto &entry : entries)
      if (entry.since == 0)
        entry.since = revision;
  };
  mark(producers);
  mark(clocks);
  mark(generations);
  mark(endpoints);
  mark(templates);
  mark(artifacts);
}
void MetadataRegistry::bind_catalog(std::shared_ptr<MetadataCatalog> catalog) {
  const auto lock = std::lock_guard{state_->mutex};
  auto &state = *state_;
  const auto first = catalog ? catalog->at(0) : std::nullopt;
  if (state.catalog_bound || !first || catalog->size() != 1 || state.rendered_revision != 1 || first->revision != 1 ||
      !same_source(first->source, state.source) || first->sha256 != state.rendered_hash)
    state.invalid("xpool semantic metadata catalog must contain its exact first complete document");
  state.mark_registered(1);
  catalog->install_coverage({state_, [](void *context, std::span<const Event> events, std::uint64_t revision) {
                               auto &owner = *static_cast<detail::MetadataState *>(context);
                               const auto held = std::lock_guard{owner.mutex};
                               owner.validate_events(events, revision);
                             }});
  state.catalog = std::move(catalog);
  state.catalog_bound = true;
}
MetadataRevision MetadataRegistry::append_revision() {
  const auto lock = std::lock_guard{state_->mutex};
  return state_->append(false);
}

MetadataRevision MetadataRegistry::ensure_revision() {
  const auto lock = std::lock_guard{state_->mutex};
  return state_->append(true);
}

MetadataRevision detail::MetadataState::append(bool only_if_changed) {
  const auto catalog = this->catalog.lock();
  if (!catalog || catalog->size() == 0)
    invalid("xpool semantic metadata has no live bound catalog");
  const auto current = *catalog->at(catalog->size() - 1);
  const auto unpublished = [](const auto &entries) {
    return std::ranges::any_of(entries, [](const auto &entry) { return entry.since == 0; });
  };
  if (only_if_changed && !unpublished(producers) && !unpublished(clocks) && !unpublished(generations) &&
      !unpublished(endpoints) && !unpublished(templates) && !unpublished(artifacts))
    return current;
  const auto previous = current.revision;
  if (previous == UINT64_MAX)
    invalid("xpool metadata revision identity exhausted");
  const auto revision = previous + 1;
  const auto bytes = render(revision);
  const auto result = catalog->append_validated(revision, bytes, this);
  mark_registered(revision);
  return result;
}

} // namespace xpool::devkit::timeline
