#include <xpool/devkit/fabric_observer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>
#include <optional>
#include <string>

#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

#include <xpool/debug/options.cuh>
#include <xpool/devkit/adapters.cuh>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/ffn_routing_observer.hpp>
#include <xpool/devkit/timeline/fabricrecording.cuh>
#include <xpool/devkit/timeline/fabricruntime.hpp>
#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/devkit/timeline/service.hpp>
#include <xpool/devkit/timeline/transportruntime.hpp>
#include <xpool/hooks.cuh>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {
namespace {
XPOOL_DEVICE_CONST FabricRecordingView recording{};
}
XPOOL_DEVICE_FN FabricRecordingView fabric_recording_view() { return recording; }
} // namespace xpool::devkit::timeline

namespace xpool::devkit::fabric_observer {
namespace {

constexpr auto batch_capacity = std::size_t{4};
constexpr auto frame_capacity = std::size_t{256};

struct RecordingOwner {
  RecordingOwner(timeline::ProcessBudgets value, timeline::CollectionPolicy installed,
                 std::shared_ptr<timeline::MetadataRegistry> definitions, timeline::FabricRecordingConfig shape,
                 int ordinal, std::size_t atn_count, std::size_t ffn_count)
      : budgets(std::move(value)), policy(installed), metadata(std::move(definitions)), config(shape), device(ordinal),
        atnagent_count(atn_count), ffnagent_count(ffn_count),
        resource(*budgets.host, timeline::MemoryKind::Metadata, metadata->source().producer), topologies(&resource) {}
  timeline::ProcessBudgets budgets;
  timeline::CollectionPolicy policy;
  std::shared_ptr<timeline::MetadataRegistry> metadata;
  timeline::FabricRecordingConfig config;
  int device;
  std::size_t atnagent_count;
  std::size_t ffnagent_count;
  timeline::BudgetMemoryResource resource;
  std::pmr::vector<ModelTopology> topologies;
  std::shared_ptr<timeline::MetadataCatalog> catalog;
  std::shared_ptr<timeline::DeviceChannel> channel;
  std::shared_ptr<timeline::LegacyRegistry> registry;
  std::shared_ptr<timeline::LegacyView> legacy;
  std::shared_ptr<timeline::SnapshotExport> exporter;
  std::shared_ptr<timeline::CollectionService> service;
  timeline::MemoryBudget::Reservation routing_charge;
  std::optional<timeline::CollectionErrorCode> read_failure;
  bool installed = false;
  bool device_retired = false;
};

std::mutex state_mutex;
std::shared_ptr<RecordingOwner> host_state;
timeline::SnapshotExportReport last_export;
timeline::FabricProgress last_progress;

std::chrono::steady_clock::time_point close_deadline(const RecordingOwner &owner) {
  return timeline::ProcessPreparation::singleton().begin_retirement(
      std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms});
}

timeline::SnapshotExportReport publication_report(const RecordingOwner &owner) {
  auto report = owner.exporter->report();
  if (!report.error && owner.service->failed()) {
    try {
      owner.service->rethrow_failure();
    } catch (const timeline::CollectionError &error) {
      report.error = error.code();
    }
  }
  if (!report.error)
    report.error = owner.read_failure;
  if (report.error == timeline::CollectionErrorCode::Transfer)
    report.error = timeline::CollectionErrorCode::Io;
  return report;
}

std::size_t recording_bytes(const timeline::CollectionPolicy &policy, timeline::FabricRecordingConfig config) {
  const auto source = timeline::DeviceSourceState::layout(policy.limits.device_event_slot_capacity);
  const auto frame = timeline::TransferLayout::for_capacity(frame_capacity);
  const auto current = timeline::FabricRecordingLayout::for_config(config);
  return xpool::utils::checked::sum(
      source.total_bytes, frame.device_bytes, current.total_bytes, sizeof(timeline::FabricRecordingView),
      policy.transport_events() && config.atnagent ? sizeof(timeline::TransportRecordingView) : 0,
      timeline::transport_recording_bytes(policy.transport_events() && config.atnagent ? policy.limits.endpoint_capacity
                                                                                       : 0));
}

std::size_t snapshot_copy_bytes(std::size_t capacity, std::size_t models) {
  return xpool::utils::checked::sum(
      sizeof(Snapshot), timeline::MemoryBudget::array_bytes(capacity, sizeof(Record), alignof(std::max_align_t)),
      timeline::MemoryBudget::array_bytes(models, sizeof(ModelTopology), alignof(std::max_align_t)));
}

std::shared_ptr<timeline::SnapshotRead> read_image(RecordingOwner &owner) {
  const auto deadline = timeline::ProcessPreparation::singleton().retirement_deadline(
      std::chrono::steady_clock::now() + std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms});
  return owner.service->read_fabric(owner.legacy, deadline);
}

void check_binding(cudaError_t error, RecordingOwner &owner, const char *operation) {
  if (error != cudaSuccess) {
    owner.channel->collector().fail_transfer();
    throw timeline::CollectionError(timeline::CollectionErrorCode::Transfer, owner.metadata->source().producer,
                                    operation, std::nullopt, cudaGetErrorString(error));
  }
}
} // namespace

void prepare(std::shared_ptr<timeline::MetadataRegistry> metadata, std::uint64_t generation,
             const xpool::fabric::ArenaProjection &projection, int pe, int device, std::string_view outdir) {
  auto &preparation = timeline::ProcessPreparation::singleton();
  const auto policy = preparation.read();
  if (!policy || !policy->fabric_events())
    return;
  const auto lock = std::lock_guard{state_mutex};
  if (host_state || !metadata || device < 0 || outdir.empty() || outdir.front() != '/' || outdir.size() > 4096 ||
      outdir.find('\0') != outdir.npos)
    throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, 0,
                                    "xpool Fabric preparation requires one actual owner and absolute output path");
  projection.validate();
  metadata->validate_preparation(device);
  metadata->require_fabric(generation, projection, pe);
  const auto config =
      timeline::FabricRecordingConfig{.generation = generation,
                                      .pe = pe,
                                      .instances = projection.instances.size(),
                                      .lanes = projection.executor_lane_count,
                                      .record_capacity = policy->legacy.fabric_observer.record_capacity,
                                      .atnagent = static_cast<std::size_t>(pe) < projection.atnagent_count,
                                      .coordinator = pe == projection.coordinator_pe(),
                                      .fifo = projection.scheduler.type() == xpool::fabric::SchedulerPolicy::Fifo};
  auto routing_elements = std::size_t{0};
  if (!config.atnagent && policy->legacy.ffn_routing_observer.enable) {
    const auto rank = static_cast<std::size_t>(pe) - projection.atnagent_count;
    for (const auto &instance : projection.instances)
      for (const auto &layer : instance.layers)
        if (layer.kind == xpool::ffn::LayerKind::Moe && !layer.ffnagent_indices.empty() &&
            layer.ffnagent_indices.front() == rank)
          routing_elements =
              std::max(routing_elements, xpool::utils::checked::prod(std::max(instance.decode_payload_row_capacity,
                                                                              instance.prefill_payload_row_capacity),
                                                                     layer.effective_topk));
  }
  const auto routing_bytes = xpool::devkit::ffn_routing_observer::allocation_bytes(routing_elements);
  auto budgets = preparation.fabric_budgets(
      xpool::utils::checked::sum(recording_bytes(*policy, config), routing_bytes), metadata->source().producer,
      sizeof(state_mutex) + sizeof(host_state) + sizeof(last_export) + sizeof(last_progress),
      sizeof(timeline::FabricRecordingView));
  auto owner = timeline::make_budgeted_shared<RecordingOwner>(
      budgets.host, timeline::MemoryKind::Control, metadata->source().producer, budgets, *policy, metadata, config,
      device, projection.atnagent_count, projection.ffnagent_count);
  owner->routing_charge =
      budgets.device->reserve(timeline::MemoryKind::Events, routing_bytes, metadata->source().producer);
  // Temporary path/string construction overlaps every cold owner. All retained
  // paths/topologies are copied into the existing budgeted Native owners.
  const auto path_peak = budgets.host->reserve(timeline::MemoryKind::Metadata, 8 * (outdir.size() + 256));
  owner->topologies.reserve(projection.instances.size());
  for (const auto &instance : projection.instances)
    owner->topologies.push_back({instance.atn_tp_size, instance.atn_dp_size});
  auto generation_text = std::array<char, 33>{};
  std::snprintf(generation_text.data(), generation_text.size(), "%016llx%016llx",
                static_cast<unsigned long long>(projection.generation_high),
                static_cast<unsigned long long>(projection.generation_low));
  const auto run = timeline::format_identity_uuid(metadata->source().run_id);
  const auto raw_path = std::filesystem::path{outdir} / ("timeline." + std::string{run.data(), run.size()});
  const auto snapshot_path =
      std::filesystem::path{outdir} /
      ("xpool.fabric-observer." + std::string{generation_text.data()} + "." + std::to_string(pe) + ".json");
  owner->catalog = timeline::MetadataCatalog::create(budgets.host, metadata->source());
  const auto first = owner->catalog->append(1, metadata->document(1));
  metadata->bind_catalog(owner->catalog);
  auto info = timeline::BatchInfo{.metadata_revision = 1};
  std::copy_n(first.sha256.begin(), info.metadata_sha256_prefix.size(), info.metadata_sha256_prefix.begin());
  owner->channel = timeline::DeviceChannel::prepare(
      budgets.host, budgets.device, metadata->source(), info, device, policy->limits.device_event_slot_capacity,
      batch_capacity, frame_capacity, config,
      policy->transport_events() && config.atnagent ? policy->limits.endpoint_capacity : 0);
  const auto legacy_slots =
      policy->transport_events()
          ? xpool::utils::checked::sum(static_cast<std::size_t>(policy->limits.endpoint_capacity), std::size_t{1})
          : std::size_t{1};
  owner->registry = timeline::LegacyRegistry::create(budgets.host, metadata->source(), legacy_slots);
  const auto scope = timeline::detail::LegacyScope{.source = metadata->source(),
                                                   .family = timeline::detail::LegacyFamily::Fabric,
                                                   .generation = generation,
                                                   .scheduler = config.fifo ? timeline::detail::LegacyPolicy::Fifo
                                                                            : timeline::detail::LegacyPolicy::Random};
  const auto citation = timeline::EvidenceMetadataSource::make(
      metadata->source().producer, 1, "metadata." + std::to_string(metadata->source().producer) + ".1.json");
  const auto all_dense = std::ranges::all_of(projection.instances, [](const auto &instance) {
    return std::ranges::all_of(instance.layers,
                               [](const auto &layer) { return layer.kind == xpool::ffn::LayerKind::Dense; });
  });
  const auto snapshot_metadata =
      timeline::SnapshotMetadata{.scheduler = citation,
                                 .routing_not_applicable = all_dense,
                                 .routing = all_dense ? citation : timeline::EvidenceMetadataSource{},
                                 .ffnagent_count = projection.ffnagent_count,
                                 .topology = citation};
  owner->legacy = owner->registry->add(scope, config.record_capacity, snapshot_metadata);
  owner->exporter = timeline::SnapshotExport::create(
      budgets.host, budgets.disk, snapshot_path.native(), scope, config.record_capacity,
      timeline::FabricJsonMetadata{generation_text.data(), static_cast<std::uint64_t>(pe), projection.atnagent_count,
                                   projection.ffnagent_count, owner->topologies});
  owner->service = preparation.collection_service();
  {
    // Original getter/copy peak coexists with one internal read/report and the
    // retained read/control owned by the single automatic-export mailbox.
    const auto read_peak = budgets.host->reserve(
        timeline::MemoryKind::Snapshot,
        xpool::utils::checked::prod(std::size_t{2}, snapshot_copy_bytes(config.record_capacity, config.instances)));
    const auto retained_read = owner->legacy->read();
    const auto exporters = std::array{owner->exporter};
    owner->service->add_device(owner->channel, info, owner->catalog, raw_path, policy->limits.chunk_bytes,
                               owner->registry, exporters);
  }
  // All Source owners are admitted before publication/activation. A cold
  // service activates at join; production binding always waits for that join.
  host_state = owner;
  last_export = {};
  last_progress = {};
}

std::size_t allocation_bytes(std::size_t instances, std::size_t lanes) {
  const auto policy = timeline::ProcessPreparation::singleton().read();
  if (!policy || !policy->fabric_events())
    return 0;
  // Include both real shapes: AtnAgent can also own the Transport current table.
  auto shape = timeline::FabricRecordingConfig{.generation = 1,
                                               .pe = 0,
                                               .instances = instances,
                                               .lanes = lanes,
                                               .record_capacity = policy->legacy.fabric_observer.record_capacity,
                                               .coordinator = true};
  const auto ffn = recording_bytes(*policy, shape);
  shape.coordinator = false;
  shape.atnagent = true;
  return std::max(ffn, recording_bytes(*policy, shape));
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FabricJoinPostEvent, HostAdapter::observe, context) {
  const auto policy = timeline::ProcessPreparation::singleton().read();
  if (!policy || !policy->fabric_events())
    return;
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || host_state->installed || host_state->device != context.device ||
      host_state->config.pe != context.pe || host_state->config.instances != context.layout.instance_count ||
      host_state->config.lanes != context.layout.executor_lane_count)
    throw timeline::CollectionError(timeline::CollectionErrorCode::MetadataUnavailable, 0,
                                    "xpool Fabric join has no matching prepared recording owner");
  auto &owner = *host_state;
  owner.metadata->require_fabric(owner.config.generation, context.projection, context.pe);
  try {
    if (!owner.channel->activation_attempted())
      owner.channel->activate();
  } catch (...) {
    if (owner.channel->retired())
      host_state.reset();
    throw;
  }
  const auto view = owner.channel->fabric_view();
  const auto device_guard = c10::cuda::CUDAGuard{context.device};
  check_binding(cudaMemcpyToSymbol(timeline::recording, &view, sizeof(view)), owner,
                "xpool Fabric view installation failed");
  owner.installed = true;
  xpool::devkit::transport_observer::activate_borrowed();
  owner.service->start();
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FabricFinalizePreEvent, HostAdapter::observe, context) {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state)
    return;
  auto &owner = *host_state;
  if (owner.device_retired)
    return;
  const auto wait = std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms};
  const auto deadline = close_deadline(owner);
  const auto device_guard = c10::cuda::CUDAGuard{context.device};
  // Native finalize reaches this only after completed local drain and actual
  // destruction of every installed FFN Graph. Stop alone is never this proof.
  xpool::devkit::transport_observer::retire_borrowed();
  const auto empty = timeline::FabricRecordingView{};
  check_binding(cudaMemcpyToSymbol(timeline::recording, &empty, sizeof(empty)), owner,
                "xpool Fabric view removal failed");
  owner.installed = false;
  if (!owner.service->retire_sources(deadline, true, wait))
    throw timeline::CollectionError(timeline::CollectionErrorCode::Io, owner.metadata->source().producer,
                                    "xpool Fabric collection retirement exceeded its original deadline");
  // Only Host storage remains. Writer failure or a blocked file operation must
  // not hold the already-drained production NVSHMEM world or its module alive.
  // The process owner remains retained until finish_retirement joins both lanes.
  owner.device_retired = true;
}

XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricAtnAgentProtocolEvent, DeviceAdapter::prepare, context) {
  const auto view = timeline::fabric_recording_view();
  if (!timeline::detail::fabric_installed(view))
    return;
  if (!context.submission || context.submission->key.instance_index != context.instance_index) {
    timeline::record_fabric_diagnostic(view, timeline::FabricDiagnostic::InvalidFacts);
    return;
  }
  timeline::prepare_fabric_atnagent(view, *context.submission);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricAtnAgentProtocolEvent, DeviceAdapter::observe, context) {
  const auto view = timeline::fabric_recording_view();
  if (timeline::detail::fabric_installed(view))
    timeline::record_fabric_atn_event(view, context.instance_index, context.kind, timeline::fabric_recording_time(),
                                      context.admission, context.commit);
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricCoordinatorProtocolEvent, DeviceAdapter::observe, context) {
  const auto view = timeline::fabric_recording_view();
  if (timeline::detail::fabric_installed(view))
    timeline::record_fabric_coordinator_event(view, context.instance_index, context.kind, context.invocation,
                                              context.ready_ticket, context.executor_lane_index,
                                              context.executor_lease_sequence, timeline::fabric_recording_time());
}
XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, DeviceAdapter::observe, context) {
  if (cooperative_groups::this_thread_block().thread_rank() != 0)
    return;
  const auto view = timeline::fabric_recording_view();
  if (timeline::detail::fabric_installed(view))
    timeline::record_fabric_ffn_event(view, context.executor_lane_index, context.kind, context.execution,
                                      context.payload_row_capacity, context.delivery,
                                      timeline::fabric_recording_time());
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || !host_state->installed)
    return std::nullopt;
  auto &owner = *host_state;
  const auto image = read_image(owner);
  auto result = Snapshot{};
  result.storage = Snapshot::Storage{owner.budgets.host,
                                     snapshot_copy_bytes(image->fabric_records().size(), owner.topologies.size())};
  result.pe = owner.config.pe;
  result.atnagent_count = owner.atnagent_count;
  result.ffnagent_count = owner.ffnagent_count;
  result.model_topologies.assign(owner.topologies.begin(), owner.topologies.end());
  result.sequence = *image->evidence()->sequence;
  result.dropped = *image->evidence()->dropped;
  result.records.assign(image->fabric_records().begin(), image->fabric_records().end());
  return result;
}

void export_snapshot() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state || !host_state->installed)
    return;
  // Automatic export follows Drain, immediately before finalization. Its read and final worker
  // join consume one bounded observation deadline; retries never restart it.
  (void)close_deadline(*host_state);
  try {
    const auto image = read_image(*host_state);
    if (!host_state->exporter->submit(image))
      throw timeline::CollectionError(timeline::CollectionErrorCode::Io, host_state->metadata->source().producer,
                                      "xpool Fabric snapshot mailbox is busy or closed");
  } catch (const timeline::CollectionError &error) {
    if (!host_state->read_failure)
      host_state->read_failure = error.code();
    throw;
  } catch (const timeline::detail::ReconstructionError &) {
    if (!host_state->read_failure)
      host_state->read_failure = timeline::CollectionErrorCode::MetadataUnavailable;
    throw;
  } catch (const std::bad_alloc &) {
    if (!host_state->read_failure)
      host_state->read_failure = timeline::CollectionErrorCode::Allocation;
    throw;
  } catch (...) {
    if (!host_state->read_failure)
      host_state->read_failure = timeline::CollectionErrorCode::Io;
    throw;
  }
}

bool cancel_preparation() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state)
    return true;
  if (host_state->channel->activation_attempted())
    return false;
  host_state.reset();
  return true;
}

bool finish_retirement() {
  const auto lock = std::lock_guard{state_mutex};
  if (!host_state)
    return true;
  auto &owner = *host_state;
  if (!owner.device_retired)
    return false;
  const auto wait = std::chrono::milliseconds{owner.policy.limits.shutdown_wait_ms};
  if (!owner.service->shutdown(close_deadline(owner), true, wait))
    return false;
  last_export = publication_report(owner);
  last_progress = owner.channel->collector().fabric_progress();
  host_state.reset();
  return true;
}

timeline::FabricProgress recording_progress() {
  const auto lock = std::lock_guard{state_mutex};
  return host_state ? host_state->channel->collector().fabric_progress() : last_progress;
}

timeline::SnapshotExportReport export_report() {
  const auto lock = std::lock_guard{state_mutex};
  return host_state ? publication_report(*host_state) : last_export;
}
} // namespace xpool::devkit::fabric_observer
