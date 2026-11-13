#include <xpool/devkit/transport_observer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory_resource>
#include <mutex>

#include <c10/cuda/CUDAGuard.h>

#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/devkit/timeline/service.hpp>
#include <xpool/devkit/timeline/transportrecording.cuh>
#include <xpool/devkit/timeline/transportruntime.hpp>
#include <xpool/hooks.cuh>
#include <xpool/transport/hooks.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::transport_observer {
namespace {

constexpr auto batch_capacity = std::size_t{4};
constexpr auto frame_capacity = std::size_t{256};
XPOOL_DEVICE_CONST timeline::TransportRecordingView recording{};

struct Endpoint {
  timeline::TransportRecordingEndpoint initial{};
  std::shared_ptr<timeline::LegacyView> legacy;
  std::shared_ptr<timeline::SnapshotExport> exporter;
  bool open = false;
  bool exported = false;
  // A CUDA error cannot retire this D2H target or its owner charge.
  timeline::TransportRecordingEndpoint copied{};
};
struct RecordingOwner {
  RecordingOwner(timeline::ProcessBudgets value, timeline::CollectionPolicy installed,
                 std::shared_ptr<timeline::MetadataRegistry> definitions, int ordinal)
      : budgets(std::move(value)), policy(installed), metadata(std::move(definitions)), device(ordinal),
        resource(*budgets.host, timeline::MemoryKind::Control, metadata->source().producer), endpoints(&resource) {
    endpoints.resize(policy.limits.endpoint_capacity);
  }
  timeline::ProcessBudgets budgets;
  timeline::CollectionPolicy policy;
  std::shared_ptr<timeline::MetadataRegistry> metadata;
  int device;
  timeline::BudgetMemoryResource resource;
  std::pmr::vector<Endpoint> endpoints;
  std::shared_ptr<timeline::CollectionService> service;
  std::shared_ptr<timeline::DeviceChannel> channel;
  std::shared_ptr<timeline::MetadataCatalog> catalog;
  std::shared_ptr<timeline::LegacyRegistry> registry;
  timeline::BatchInfo initial_metadata{};
  bool registered = false;
  timeline::MemoryBudget::Reservation binding_host;
  timeline::MemoryBudget::Reservation binding_device;
  bool borrowed = false;
  bool installed = false;
  bool uncertain = false;
  std::optional<timeline::CollectionErrorCode> failure;
  std::array<char, 512> diagnostic{};
};
std::mutex state_mutex;
std::shared_ptr<RecordingOwner> host_state;

[[noreturn]] void invalid(std::uint64_t producer, const char *message) {
  throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, producer, message);
}
void check(cudaError_t status, RecordingOwner &owner, const char *message) {
  if (status != cudaSuccess) {
    owner.uncertain = true;
    owner.channel->collector().fail_transfer();
    owner.failure = timeline::CollectionErrorCode::Io;
    if (owner.diagnostic.front() == '\0')
      std::snprintf(owner.diagnostic.data(), owner.diagnostic.size(), "%s: %s", message, cudaGetErrorString(status));
    throw timeline::CollectionError(timeline::CollectionErrorCode::Transfer, owner.metadata->source().producer, message,
                                    std::nullopt, cudaGetErrorString(status));
  }
}
[[noreturn]] void uncertain(const RecordingOwner &owner, const char *message) {
  throw timeline::CollectionError(timeline::CollectionErrorCode::Io, owner.metadata->source().producer, message,
                                  std::nullopt, owner.diagnostic.data());
}
auto deadline(const RecordingOwner &owner, bool closing = false) {
  auto &preparation = timeline::ProcessPreparation::singleton();
  const auto wait = std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms};
  return closing ? preparation.begin_retirement(wait)
                 : preparation.retirement_deadline(std::chrono::steady_clock::now() + wait);
}
std::size_t copy_bytes(std::size_t count) {
  return timeline::MemoryBudget::array_bytes(count, sizeof(Record), alignof(std::max_align_t));
}
void install(RecordingOwner &owner) {
  if (owner.installed)
    return;
  if (owner.uncertain || owner.channel->retired())
    invalid(owner.metadata->source().producer, "xpool Transport activation cannot reuse an uncertain Source");
  const auto view = owner.channel->transport_view();
  if (!view.endpoints || view.capacity != owner.endpoints.size())
    invalid(owner.metadata->source().producer, "xpool Transport Source lacks its admitted Endpoint table");
  const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
  for (auto index = std::size_t{0}; index < owner.endpoints.size(); ++index)
    if (owner.endpoints[index].legacy)
      check(cudaMemcpy(view.endpoints + index, &owner.endpoints[index].initial,
                       sizeof(timeline::TransportRecordingEndpoint), cudaMemcpyHostToDevice),
            owner, "xpool Transport Endpoint installation failed");
  check(cudaMemcpyToSymbol(recording, &view, sizeof(view)), owner, "xpool Transport binding installation failed");
  owner.installed = true;
}
std::shared_ptr<timeline::SnapshotRead> read_image(RecordingOwner &owner, std::size_t index) {
  if (!owner.installed || owner.uncertain || owner.channel->retired())
    invalid(owner.metadata->source().producer, "xpool Transport read requires its live quiescent owner");
  auto &current = owner.endpoints[index].copied;
  const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
  check(
      cudaMemcpy(&current, owner.channel->transport_view().endpoints + index, sizeof(current), cudaMemcpyDeviceToHost),
      owner, "xpool Transport quiescent count read failed");
  const auto &endpoint = owner.endpoints[index];
  if (current.endpoint_owner != endpoint.initial.endpoint_owner ||
      current.endpoint_sequence != endpoint.initial.endpoint_sequence || current.exhausted || current.diagnostics)
    invalid(owner.metadata->source().producer, "xpool Transport original count or identity evidence is unavailable");
  return owner.service->read_legacy(endpoint.legacy, current.operations, deadline(owner));
}
void publish(RecordingOwner &owner, std::size_t index) {
  auto &endpoint = owner.endpoints[index];
  if (endpoint.exported)
    return;
  auto image = read_image(owner, index);
  if (!endpoint.exporter->submit(std::move(image)))
    throw timeline::CollectionError(timeline::CollectionErrorCode::Io, owner.metadata->source().producer,
                                    "xpool Transport snapshot mailbox is busy or closed");
  endpoint.exported = true;
}
void close_export(RecordingOwner &owner, std::size_t index) {
  auto &endpoint = owner.endpoints[index];
  try {
    publish(owner, index);
    endpoint.exporter->close();
  } catch (const timeline::CollectionError &error) {
    owner.failure =
        error.code() == timeline::CollectionErrorCode::Transfer ? timeline::CollectionErrorCode::Io : error.code();
    (void)endpoint.exporter->close_unavailable(*owner.failure);
  } catch (const timeline::detail::ReconstructionError &) {
    owner.failure = timeline::CollectionErrorCode::MetadataUnavailable;
    (void)endpoint.exporter->close_unavailable(*owner.failure);
  }
}
XPOOL_DEVICE_FN timeline::TransportRecordingEndpoint *endpoint(const xpool::transport::OperationFact &operation) {
  for (auto index = std::size_t{0}; index < recording.capacity; ++index) {
    auto &candidate = recording.endpoints[index];
    if (candidate.endpoint_owner != 0 && candidate.endpoint_owner == operation.endpoint_owner &&
        candidate.endpoint_sequence == operation.endpoint_sequence && candidate.generation == operation.generation)
      return &candidate;
  }
  return nullptr;
}
} // namespace

void prepare_endpoint(std::shared_ptr<timeline::MetadataRegistry> metadata,
                      const xpool::transport::ObservationRegion &region, std::size_t instance_index, std::uint32_t rank,
                      int device, bool atnagent, std::uint64_t pid, std::string_view model_id,
                      std::string_view arena_suffix, std::string_view path) {
  auto &preparation = timeline::ProcessPreparation::singleton();
  const auto policy = preparation.read();
  if (!policy || !policy->transport_events())
    return;
  const auto lock = std::lock_guard{state_mutex};
  if (!metadata || region.version != timeline::kContractVersion || region.generation == 0 ||
      region.owner_producer == 0 || region.endpoint_sequence == 0 || device < 0 || path.empty() ||
      path.front() != '/' || path.size() > 4096 || path.find('\0') != path.npos ||
      (atnagent && region.owner_producer != metadata->source().producer))
    invalid(0, "xpool Transport preparation requires registered Endpoint facts and an absolute path");
  if (!host_state) {
    const auto budgets = preparation.budgets();
    if (!budgets || !budgets->device)
      invalid(metadata->source().producer, "xpool Transport requires the prepared shared Device budget");
    auto owner = timeline::make_budgeted_shared<RecordingOwner>(
        budgets->host, timeline::MemoryKind::Control, metadata->source().producer, *budgets, *policy, metadata, device);
    owner->binding_host =
        budgets->host->reserve(timeline::MemoryKind::Control, sizeof(state_mutex) + sizeof(host_state));
    owner->binding_device = budgets->device->reserve(timeline::MemoryKind::Control, sizeof(recording));
    owner->service = preparation.collection_service();
    if (auto shared = owner->service->device_source(metadata->source())) {
      if (shared->channel->device_index() != device)
        invalid(metadata->source().producer, "xpool shared Transport Source changed its Device mapping");
      owner->channel = shared->channel;
      owner->catalog = shared->metadata;
      owner->registry = shared->legacy;
      owner->borrowed = true;
      owner->registered = true;
      if (!owner->registry)
        invalid(metadata->source().producer, "xpool shared Transport Source has no reconstruction registry");
    } else {
      metadata->validate_preparation(device);
      owner->catalog = timeline::MetadataCatalog::create(budgets->host, metadata->source());
      const auto first = owner->catalog->append(1, metadata->document(1));
      metadata->bind_catalog(owner->catalog);
      auto info = timeline::BatchInfo{.metadata_revision = 1};
      std::copy_n(first.sha256.begin(), info.metadata_sha256_prefix.size(), info.metadata_sha256_prefix.begin());
      owner->channel = timeline::DeviceChannel::prepare(
          budgets->host, budgets->device, metadata->source(), info, device, policy->limits.device_event_slot_capacity,
          batch_capacity, frame_capacity, std::nullopt, policy->limits.endpoint_capacity);
      owner->registry =
          timeline::LegacyRegistry::create(budgets->host, metadata->source(), policy->limits.endpoint_capacity);
      owner->initial_metadata = info;
    }
    // Retain before any CUDA installation; uncertain completion cannot free it.
    host_state = std::move(owner);
  }
  auto &owner = *host_state;
  if (owner.metadata != metadata || owner.device != device || owner.uncertain)
    invalid(metadata->source().producer, "xpool Transport cannot replace its live process Source");
  (void)metadata->ensure_revision();
  const auto initial =
      timeline::TransportRecordingEndpoint{.endpoint_owner = region.owner_producer,
                                           .endpoint_sequence = region.endpoint_sequence,
                                           .generation = region.generation,
                                           .instance_index = instance_index,
                                           .instance_rank = rank,
                                           .role = atnagent ? 2U : 1U,
                                           .capacity = policy->legacy.transport_observer.record_capacity};
  auto index = owner.endpoints.size();
  for (auto slot = std::size_t{0}; slot < owner.endpoints.size(); ++slot) {
    const auto &entry = owner.endpoints[slot];
    if (entry.initial.endpoint_owner == initial.endpoint_owner &&
        entry.initial.endpoint_sequence == initial.endpoint_sequence) {
      if (!entry.legacy || entry.initial.generation != initial.generation ||
          entry.initial.instance_index != instance_index || entry.initial.instance_rank != rank ||
          entry.initial.role != initial.role || entry.exporter->path() != path)
        invalid(metadata->source().producer, "xpool Transport Endpoint retry changed its lifetime or facts");
      return;
    }
    if (entry.exporter && entry.exporter->path() == path) {
      if (entry.open)
        invalid(metadata->source().producer, "xpool Transport destination still has an open Endpoint");
      index = slot;
      break;
    }
    if (!entry.exporter && index == owner.endpoints.size())
      index = slot;
  }
  if (index == owner.endpoints.size())
    throw timeline::CollectionError(timeline::CollectionErrorCode::ResourceBudget, metadata->source().producer,
                                    "xpool Transport Endpoint table is full");
  const auto scope =
      timeline::detail::LegacyScope{.source = metadata->source(),
                                    .family = atnagent ? timeline::detail::LegacyFamily::TransportAtnAgent
                                                       : timeline::detail::LegacyFamily::TransportInstance,
                                    .generation = region.generation,
                                    .endpoint_owner = region.owner_producer,
                                    .endpoint_sequence = region.endpoint_sequence};
  auto &entry = owner.endpoints[index];
  const auto envelope = timeline::TransportJsonMetadata{pid, model_id, rank, arena_suffix};
  auto legacy = owner.registry->add(scope, initial.capacity);
  try {
    const auto peak = owner.budgets.host->reserve(timeline::MemoryKind::Snapshot, 2 * copy_bytes(initial.capacity));
    auto exporter = std::shared_ptr<timeline::SnapshotExport>{};
    if (entry.exporter) {
      // Probe reads before the no-fail destination exchange. This immutable
      // policy/path keeps the formatter geometry; replacement itself admits
      // both old and new formatter allocations before rebinding the publisher.
      owner.service->preflight();
      exporter = owner.service->replace_snapshot_export(entry.exporter, scope, initial.capacity, envelope);
    } else {
      exporter = timeline::SnapshotExport::create(owner.budgets.host, owner.budgets.disk, path, scope, initial.capacity,
                                                  envelope);
      if (owner.registered) {
        owner.service->preflight();
        owner.service->attach_snapshot_export(exporter);
      }
    }
    entry = Endpoint{initial, legacy, std::move(exporter), atnagent, false};
  } catch (...) {
    owner.registry->retire(legacy);
    throw;
  }
  if (!owner.registered) {
    const auto paths = owner.budgets.host->reserve(timeline::MemoryKind::Metadata, 8 * (path.size() + 256));
    const auto run = timeline::format_identity_uuid(metadata->source().run_id);
    const auto raw = std::filesystem::path{path}.parent_path() / ("timeline." + std::string{run.data(), run.size()});
    const auto exporters = std::array{entry.exporter};
    owner.service->add_device(owner.channel, owner.initial_metadata, owner.catalog, raw, policy->limits.chunk_bytes,
                              owner.registry, exporters);
    owner.registered = true;
    if (!owner.channel->activation_attempted()) {
      owner.service->preflight();
      owner.channel->activate();
    }
    install(owner);
    owner.service->start();
  } else if (owner.installed) {
    const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
    check(cudaMemcpy(owner.channel->transport_view().endpoints + index, &entry.initial, sizeof(entry.initial),
                     cudaMemcpyHostToDevice),
          owner, "xpool Transport replacement installation failed");
  }
}

void activate_borrowed() {
  const auto lock = std::lock_guard{state_mutex};
  if (host_state && host_state->borrowed)
    install(*host_state);
}
void retire_borrowed() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || !host_state->borrowed || !host_state->installed)
    return;
  auto &owner = *host_state;
  if (owner.uncertain)
    uncertain(owner, "xpool uncertain Transport copy retains its borrowed Source");
  (void)deadline(owner, true);
  for (auto index = std::size_t{0}; index < owner.endpoints.size(); ++index)
    if (owner.endpoints[index].open)
      close_export(owner, index);
  if (owner.uncertain)
    uncertain(owner, "xpool uncertain Transport copy prevents Source retirement");
  const auto empty = timeline::TransportRecordingView{};
  const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
  check(cudaMemcpyToSymbol(recording, &empty, sizeof(empty)), owner, "xpool Transport binding removal failed");
  owner.installed = false;
}

XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointOpenPostEvent, HostAdapter::observe, context) {
  // AtnAgent preparation follows arena creation; Instance preparation precedes attach.
  if (context.site == xpool::hooks::TransportEndpointSite::AtnAgent)
    return;
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state) {
    const auto policy = timeline::ProcessPreparation::singleton().read();
    if (policy && policy->transport_events())
      invalid(0, "xpool Instance recording requires Endpoint preparation before attach");
    return;
  }
  for (auto &entry : host_state->endpoints)
    if (entry.legacy && entry.initial.instance_index == context.layout.instance_index &&
        entry.initial.instance_rank == context.layout.instance_rank && entry.initial.role == 1) {
      entry.open = true;
      return;
    }
  invalid(host_state->metadata->source().producer, "xpool Instance attach has no prepared recording Endpoint");
}
XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointClosePreEvent, HostAdapter::observe, context) {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state)
    return;
  auto &owner = *host_state;
  for (auto index = std::size_t{0}; index < owner.endpoints.size(); ++index) {
    auto &entry = owner.endpoints[index];
    if (!entry.open || entry.initial.instance_index != context.layout.instance_index ||
        entry.initial.instance_rank != context.layout.instance_rank)
      continue;
    if (owner.installed) {
      close_export(owner, index);
      if (owner.uncertain)
        uncertain(owner, "xpool uncertain Transport copy prevents Endpoint reuse");
      const auto empty = timeline::TransportRecordingEndpoint{};
      const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
      check(
          cudaMemcpy(owner.channel->transport_view().endpoints + index, &empty, sizeof(empty), cudaMemcpyHostToDevice),
          owner, "xpool Transport Endpoint removal failed");
    } else {
      entry.exporter->close();
    }
    owner.registry->retire(entry.legacy);
    entry.legacy.reset();
    entry.open = false;
    return;
  }
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportInstanceProtocolEvent, DeviceAdapter::observe, context) {
  if (auto *owner = endpoint(context.operation))
    timeline::record_transport_event(recording, *owner, context.kind, context.operation, context.request,
                                     context.payload_rows, context.result_code, xpool::utils::time::now());
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (auto *owner = endpoint(context.operation))
    timeline::record_transport_event(recording, *owner, context.kind, context.operation, context.request,
                                     context.payload_rows, context.result_code, xpool::utils::time::now());
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || !host_state->installed)
    return std::nullopt;
  auto &owner = *host_state;
  const auto count = std::ranges::count_if(owner.endpoints, [](const auto &entry) { return entry.open; });
  if (count == 0)
    return std::nullopt;
  auto result = Snapshot{};
  result.storage = {owner.budgets.host,
                    timeline::MemoryBudget::array_bytes(count, sizeof(EndpointSnapshot), alignof(std::max_align_t))};
  result.endpoints.reserve(count);
  for (auto index = std::size_t{0}; index < owner.endpoints.size(); ++index) {
    const auto &entry = owner.endpoints[index];
    if (!entry.open)
      continue;
    const auto image = read_image(owner, index);
    auto value = EndpointSnapshot{};
    value.storage = {owner.budgets.host, copy_bytes(image->transport_records().size())};
    value.instance_index = entry.initial.instance_index;
    value.instance_rank = entry.initial.instance_rank;
    value.sequence = *image->evidence()->sequence;
    value.dropped = *image->evidence()->dropped;
    value.records.assign(image->transport_records().begin(), image->transport_records().end());
    result.endpoints.push_back(std::move(value));
  }
  std::ranges::sort(result.endpoints, {},
                    [](const auto &value) { return std::pair{value.instance_index, value.instance_rank}; });
  return result;
}
void export_snapshot() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || !host_state->installed)
    return;
  (void)deadline(*host_state, true);
  for (auto index = std::size_t{0}; index < host_state->endpoints.size(); ++index)
    if (host_state->endpoints[index].open)
      publish(*host_state, index);
}
bool finish_retirement() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state)
    return true;
  auto &owner = *host_state;
  if (owner.uncertain || std::ranges::any_of(owner.endpoints, [](const auto &entry) { return entry.open; }))
    return false;
  if (owner.borrowed) {
    if (owner.channel->activation_attempted() && !owner.channel->retired())
      return false;
  } else {
    const auto empty = timeline::TransportRecordingView{};
    const auto guard = c10::cuda::CUDAGuard{static_cast<c10::DeviceIndex>(owner.device)};
    check(cudaMemcpyToSymbol(recording, &empty, sizeof(empty)), owner, "xpool Transport binding removal failed");
    owner.installed = false;
    if (!owner.service->shutdown(deadline(owner, true), true,
                                 std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms}))
      return false;
  }
  const auto failure = owner.failure;
  const auto service = owner.service;
  host_state.reset();
  service->rethrow_failure();
  if (failure)
    throw timeline::CollectionError(*failure, 0, "xpool Transport retired with unavailable snapshot evidence");
  return true;
}
} // namespace xpool::devkit::transport_observer
