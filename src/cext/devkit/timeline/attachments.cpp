#include <xpool/devkit/timeline/attachments.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <mutex>
#include <string>

#include <c10/util/Logging.h>

#include <xpool/devkit/timeline/digest.hpp>
#include <xpool/devkit/timeline/identity.hpp>
#include <xpool/devkit/timeline/json.hpp>
#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {
namespace {
struct AttachmentOwner {
  explicit AttachmentOwner(ProcessBudgets value)
      : budgets(std::move(value)), graph_resource(*budgets.host, MemoryKind::Metadata),
        compute_templates(&graph_resource) {}
  ProcessBudgets budgets;
  MemoryBudget::Reservation routing_charge;
  MemoryBudget::Reservation routing_host_charge;
  BudgetMemoryResource graph_resource;
  std::pmr::map<std::size_t, std::uint64_t> compute_templates;
  std::shared_ptr<CollectionService> service;
  std::shared_ptr<MetadataRegistry> host;
  std::shared_ptr<MetadataRegistry> device;
  std::shared_ptr<HostChannel> host_channel;
  std::shared_ptr<DeviceChannel> device_channel;
  std::shared_ptr<MetadataCatalog> host_catalog;
  std::shared_ptr<MetadataCatalog> device_catalog;
  AttemptSequence templates;
  AttemptSequence executions;
  std::uint64_t generation;
  bool borrowed = false;
};
std::mutex owner_mutex;
std::shared_ptr<AttachmentOwner> owner;
RoutingEncoder routing_encoder = nullptr;

std::string run_text(BatchSource source) {
  const auto text = format_identity_uuid(source.run_id);
  return {text.data(), text.size()};
}
std::filesystem::path root(std::string_view outdir, BatchSource source) {
  return std::filesystem::path{outdir} / ("timeline." + run_text(source));
}
BatchInfo register_catalog(const std::shared_ptr<MetadataRegistry> &metadata,
                           const std::shared_ptr<MetadataCatalog> &catalog) {
  const auto first = catalog->append(1, metadata->document(1));
  metadata->bind_catalog(catalog);
  auto info = BatchInfo{.metadata_revision = 1};
  std::copy_n(first.sha256.begin(), info.metadata_sha256_prefix.size(), info.metadata_sha256_prefix.begin());
  return info;
}
void append_json(void *context, std::string_view text) { static_cast<std::string *>(context)->append(text); }
class GraphJob final : public ArtifactJob {
public:
  GraphJob(std::shared_ptr<AttachmentOwner> owner, GraphStructure structure, std::uint64_t sequence, bool lane,
           std::size_t index, MemoryBudget::Reservation charge)
      : owner_(std::move(owner)), charge_(std::move(charge)), structure_(std::move(structure)), sequence_(sequence),
        lane_(lane), index_(index) {}
  BatchSource source() const noexcept override { return owner_->host->source(); }
  void write(RawWriter &writer) override {
    auto bytes = std::string{};
    detail::JsonOutput out{{&bytes, append_json}};
    out.text("{\"contract_version\":7,\"run_id\":");
    out.quoted(run_text(source()));
    out.text(",\"producer_id\":");
    out.number(source().producer);
    out.text(",\"graph_owner\":");
    out.number(source().producer);
    out.text(",\"template_sequence\":");
    out.number(sequence_);
    out.text(",\"kind\":");
    out.quoted(lane_ ? "lane_service" : "primary");
    out.text(",\"index\":");
    out.number(index_);
    out.text(",\"node_counts\":{");
    auto first = true;
    for (const auto &[kind, count] : structure_.node_counts) {
      if (!first)
        out.text(",");
      first = false;
      out.quoted(std::to_string(static_cast<int>(kind)));
      out.text(":");
      out.number(count);
    }
    out.text("},\"binding_site_count\":");
    out.number(structure_.binding_site_count);
    out.text(",\"compute_branch_count\":");
    out.number(structure_.compute_branch_count);
    out.text(",\"delivery_branch_count\":");
    out.number(structure_.delivery_branch_count);
    out.text("}\n");
    out.finish();
    const auto path = "graph/" + std::to_string(source().producer) + "." + std::to_string(sequence_) + ".json";
    writer.publish_attachment(path, std::as_bytes(std::span{bytes}), false, EvidenceClass::Graph);
    owner_->host->add_artifact(sequence_, "graph_structure", MetadataArtifactStatus::Published, path,
                               sha256(std::as_bytes(std::span{bytes})));
    owner_->host->ensure_revision();
    writer.publish_pending_metadata(*owner_->host_catalog);
  }

private:
  std::shared_ptr<AttachmentOwner> owner_;
  MemoryBudget::Reservation charge_;
  GraphStructure structure_;
  std::uint64_t sequence_;
  bool lane_;
  std::size_t index_;
};
class RoutingJob final : public ArtifactJob {
public:
  RoutingJob(std::shared_ptr<AttachmentOwner> owner, std::vector<RoutingAttachment> records,
             MemoryBudget::Reservation charge)
      : owner_(std::move(owner)), charge_(std::move(charge)), records_(std::move(records)) {}
  BatchSource source() const noexcept override { return owner_->device->source(); }
  std::size_t routing_count() const noexcept override { return records_.size(); }
  void write(RawWriter &writer) override {
    const auto path = "routing/" + std::to_string(source().producer) + ".safetensors";
    auto hash = std::array<std::byte, 32>{};
    auto published = false;
    auto failure = std::exception_ptr{};
    try {
      if (!routing_encoder)
        throw CollectionError(CollectionErrorCode::MetadataUnavailable, source().producer,
                              "Safetensors encoder is not installed");
      const auto bytes = routing_encoder(source(), records_);
      writer.publish_attachment(path, std::as_bytes(std::span{bytes}));
      hash = sha256(std::as_bytes(std::span{bytes}));
      published = true;
    } catch (...) {
      failure = std::current_exception();
    }
    try {
      owner_->device->add_artifact(
          1, "routing", published ? MetadataArtifactStatus::Published : MetadataArtifactStatus::PublishFailed,
          published ? std::optional<std::string_view>{path} : std::nullopt,
          published ? std::optional{hash} : std::nullopt);
      owner_->device->ensure_revision();
      writer.publish_pending_metadata(*owner_->device_catalog);
    } catch (const std::exception &secondary) {
      if (failure) {
        LOG(ERROR) << "xpool Routing metadata publication also failed: "
                   << std::string_view{secondary.what()}.substr(0, 512);
        std::rethrow_exception(failure);
      }
      failure = std::current_exception();
    }
    // Payload publication and closing index outcomes are distinct. A failed
    // payload retains its explicit outcome; an unpublished index stays unknown.
    auto index = std::string{};
    detail::JsonOutput out{{&index, append_json}};
    out.text("{\"contract_version\":7,\"run_id\":");
    out.quoted(run_text(source()));
    out.text(",\"producer_id\":");
    out.number(source().producer);
    out.text(",\"complete\":true,\"attachments\":[");
    auto first = true;
    for (std::size_t i = 0; i < records_.size(); ++i) {
      const auto &record = records_[i];
      if (!first)
        out.text(",");
      first = false;
      out.text("{\"owner\":");
      out.number(source().producer);
      out.text(",\"sequence\":");
      out.number(record.sequence);
      out.text(published ? ",\"status\":\"published\",\"path\":" : ",\"status\":\"publish_failed\",\"path\":");
      if (published)
        out.quoted(path);
      else
        out.text("null");
      out.text(",\"sha256\":");
      if (published)
        out.text("\"");
      constexpr char digits[] = "0123456789abcdef";
      if (published)
        for (auto byte : hash) {
          const auto v = std::to_integer<unsigned>(byte);
          out.text(std::string{digits[v >> 4], digits[v & 15]});
        }
      out.text(published ? "\",\"tensor_keys\":[" : "null,\"tensor_keys\":[");
      out.quoted("records." + std::to_string(i) + ".topk_ids");
      out.text(",");
      out.quoted("records." + std::to_string(i) + ".topk_weights");
      out.text("],\"generation\":");
      out.number(record.generation);
      out.text(",\"instance_index\":");
      out.number(record.instance_index);
      out.text(",\"invocation_sequence\":");
      out.number(record.invocation_sequence);
      out.text(",\"lane_index\":");
      out.number(record.lane_index);
      out.text(",\"lease_sequence\":");
      out.number(record.lease_sequence);
      out.text(",\"layer_ordinal\":");
      out.number(record.layer_ordinal);
      out.text(",\"publisher\":");
      out.number(source().producer);
      out.text(",\"pe\":");
      out.number(record.pe);
      out.text(",\"clock_domain_id\":");
      out.number(record.clock_domain_id);
      out.text(",\"raw_publication_time_ns\":");
      out.number(record.raw_publication_time_ns);
      out.text(",\"shape\":[");
      out.number(record.topk_ids.size(0));
      out.text(",");
      out.number(record.topk_ids.size(1));
      out.text("],\"dtypes\":[\"int32\",\"float32\"]}");
    }
    out.text("]}\n");
    out.finish();
    try {
      writer.publish_attachment("attachments." + std::to_string(source().producer) + ".json",
                                std::as_bytes(std::span{index}), true);
    } catch (const std::exception &secondary) {
      if (failure) {
        LOG(ERROR) << "xpool Routing closing index also failed: " << std::string_view{secondary.what()}.substr(0, 512);
        std::rethrow_exception(failure);
      }
      throw;
    }
    if (failure)
      std::rethrow_exception(failure);
  }

private:
  std::shared_ptr<AttachmentOwner> owner_;
  MemoryBudget::Reservation charge_;
  std::vector<RoutingAttachment> records_;
};
} // namespace

void install_routing_encoder(RoutingEncoder encoder) { routing_encoder = encoder; }
void prepare_attachments(std::shared_ptr<MetadataRegistry> host, std::shared_ptr<MetadataRegistry> device, int ordinal,
                         std::uint64_t generation, std::string_view graph_outdir, std::string_view routing_outdir,
                         xpool::devkit::ffn_routing_observer::MemoryDemand routing_demand) {
  const auto lock = std::lock_guard{owner_mutex};
  if (owner) {
    if (owner->host != host || owner->device != device)
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "attachment owner cannot change");
    return;
  }
  auto &preparation = ProcessPreparation::singleton();
  const auto budgets = preparation.budgets();
  const auto policy = preparation.read();
  if (!budgets || !policy)
    return;
  const auto role = host->process_role();
  const auto graph_enabled = policy->graph_events() && role != xpool::RuntimeRole::AtnAgent;
  const auto routing_enabled = policy->routing_events() && role == xpool::RuntimeRole::FfnAgent &&
                               device->capabilities().routing == MetadataCapability::Enabled;
  const auto metadata_only =
      policy->routing_events() && role == xpool::RuntimeRole::FfnAgent && !routing_enabled && !graph_enabled;
  if (!graph_enabled && !routing_enabled && !metadata_only)
    return;
  auto current =
      make_budgeted_shared<AttachmentOwner>(budgets->host, MemoryKind::Control, host->source().producer, *budgets);
  current->host = host;
  current->device = device;
  current->generation = generation;
  current->service = preparation.collection_service();
  if (auto shared = current->service->device_source(device->source())) {
    current->device_channel = shared->channel;
    current->device_catalog = shared->metadata;
    current->borrowed = true;
  }
  if (graph_enabled || metadata_only) {
    current->host_catalog = MetadataCatalog::create(budgets->host, host->source());
    const auto info = register_catalog(host, current->host_catalog);
    current->host_channel = HostChannel::create(budgets->host, host->source(), info, 256, 4, 256);
    current->service->add_host(current->host_channel, current->host_catalog,
                               root(metadata_only ? routing_outdir : graph_outdir, host->source()),
                               policy->limits.chunk_bytes);
    current->service->register_directory(host->source(), metadata_only ? EvidenceClass::Routing : EvidenceClass::Graph,
                                         root(metadata_only ? routing_outdir : graph_outdir, host->source()));
  }
  if (routing_enabled) {
    if (routing_demand.device_bytes == 0 || routing_demand.host_bytes == 0)
      throw CollectionError(CollectionErrorCode::ResourceBudget, device->source().producer,
                            "Routing allocation demand is unknown");
    current->routing_host_charge =
        budgets->host->reserve(MemoryKind::Writer, routing_demand.host_bytes, device->source().producer);
    if (auto shared = current->service->device_source(device->source())) {
      current->device_channel = shared->channel;
      current->device_catalog = shared->metadata;
      current->borrowed = true;
      current->routing_charge =
          current->budgets.device->reserve(MemoryKind::Events, routing_demand.device_bytes, device->source().producer);
    } else {
      auto device_budget =
          budgets->device ? budgets->device
                          : make_budgeted_shared<MemoryBudget>(budgets->host, MemoryKind::Control,
                                                               device->source().producer, policy->limits.device_bytes);
      const auto source_layout = DeviceSourceState::layout(policy->limits.device_event_slot_capacity);
      const auto transfer = TransferLayout::for_capacity(256);
      const auto required =
          xpool::utils::checked::sum(source_layout.total_bytes, transfer.device_bytes, routing_demand.device_bytes);
      if (required > policy->limits.device_bytes)
        throw CollectionError(CollectionErrorCode::ResourceBudget, device->source().producer,
                              "Complete Routing/Event/transfer demand exceeds Device budget");
      current->routing_charge =
          device_budget->reserve(MemoryKind::Events, routing_demand.device_bytes, device->source().producer);
      current->device_catalog = MetadataCatalog::create(budgets->host, device->source());
      const auto info = register_catalog(device, current->device_catalog);
      current->budgets.device = device_budget;
      current->device_channel = DeviceChannel::prepare(budgets->host, device_budget, device->source(), info, ordinal,
                                                       policy->limits.device_event_slot_capacity, 4, 256);
      current->service->add_device(current->device_channel, info, current->device_catalog,
                                   root(routing_outdir, device->source()), policy->limits.chunk_bytes);
      current->device_channel->activate();
    }
  }
  if (routing_enabled)
    current->service->register_directory(device->source(), EvidenceClass::Routing,
                                         root(routing_outdir, device->source()));
  owner = std::move(current);
  if (!owner->borrowed)
    owner->service->start();
}
GraphStructure make_graph_structure() {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Graph owner is not prepared");
  return GraphStructure{&owner->graph_resource};
}
void submit_graph_structure(GraphStructure structure, bool lane, std::size_t index) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->host_channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Graph Source is not prepared");
  const auto id = owner->templates.try_next();
  if (!id)
    throw CollectionError(CollectionErrorCode::ResourceBudget, owner->host->source().producer,
                          "Graph template identities exhausted");
  owner->host->add_template(owner->host->source().producer, id, lane ? "lane_service" : "primary", std::nullopt,
                            lane ? std::nullopt : std::optional<std::uint64_t>{index},
                            lane ? std::optional<std::uint64_t>{index} : std::nullopt);
  if (lane) {
    if (structure.compute_branch_count != owner->compute_templates.size())
      throw CollectionError(CollectionErrorCode::MetadataUnavailable, owner->host->source().producer,
                            "Lane selectable compute inventory disagrees with registered templates");
    for (const auto &[signature, child] : owner->compute_templates) {
      (void)signature;
      owner->host->add_compute_template(owner->host->source().producer, id, child);
    }
  } else {
    owner->compute_templates[index] = id;
  }
  owner->host->ensure_revision();
  auto event = Event{.clock_domain_id = owner->host->source().clock_domain_id,
                     .kind = 503,
                     .presence = 1U << 3,
                     .graph_owner = owner->host->source().producer,
                     .graph_template = id,
                     .role = 4,
                     .data0 = lane ? 3U : 1U,
                     .data1 = 3};
  timespec timestamp{};
  if (::clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    throw CollectionError(CollectionErrorCode::Io, owner->host->source().producer, "Graph clock read failed");
  event.raw_time_ns =
      static_cast<std::uint64_t>(timestamp.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(timestamp.tv_nsec);
  const auto reserved = try_reserve_host(owner->host_channel->buffer());
  if (reserved.status == ReserveStatus::Reserved)
    commit_host(owner->host_channel->buffer(), reserved.token, event);
  const auto bytes = xpool::utils::checked::sum(
      std::size_t{16384}, xpool::utils::checked::prod(structure.node_counts.size(), std::size_t{256}));
  auto charge = owner->budgets.host->reserve(MemoryKind::Writer, bytes);
  owner->service->submit_artifact(
      make_budgeted_shared<GraphJob>(owner->budgets.host, MemoryKind::Control, owner->host->source().producer, owner,
                                     std::move(structure), id, lane, index, std::move(charge)));
}
RoutingSource routing_source() {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->device_channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing Source is not prepared");
  const auto pe = owner->device->fabric_pe(owner->generation, owner->device->source().producer);
  if (!pe)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, owner->device->source().producer,
                          "Routing Source is not a Fabric member");
  return {owner->device_channel->buffer_view(), owner->generation, *pe};
}
BufferView routing_event_buffer() {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->device_channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing Source is not prepared");
  return owner->device_channel->buffer_view();
}
void record_routing_counts(std::uint64_t attempted, std::uint64_t full, std::uint64_t oversized) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->device_channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing counters have no Source");
  auto &collector = owner->device_channel->collector();
  collector.record(ExternalCounter::AttachmentAttempted, attempted);
  collector.record(ExternalCounter::AttachmentCapacityExhausted, full);
  collector.record(ExternalCounter::AttachmentPayloadOversize, oversized);
}
[[noreturn]] void fail_routing_transfer(std::string_view operation, std::string_view cause) {
  const auto lock = std::lock_guard{owner_mutex};
  if (owner && owner->device_channel)
    owner->device_channel->collector().fail_transfer();
  throw CollectionError(CollectionErrorCode::Transfer, owner ? owner->device->source().producer : 0,
                        std::string{operation}, std::nullopt, std::string{cause});
}
MemoryBudget::Reservation admit_routing_device(std::size_t bytes) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->budgets.device)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing Device budget is not prepared");
  if (owner->routing_charge.bytes() != bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, owner->device->source().producer,
                          "Routing allocation differs from complete preparation demand");
  return std::move(owner->routing_charge);
}
MemoryBudget::Reservation admit_routing_copy(std::size_t bytes) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing owner is not prepared");
  if (owner->routing_host_charge.bytes() != bytes)
    throw CollectionError(CollectionErrorCode::ResourceBudget, owner->device->source().producer,
                          "Routing Host copy differs from admitted coexistence peak");
  return std::move(owner->routing_host_charge);
}
void submit_routing_attachments(std::vector<RoutingAttachment> records, MemoryBudget::Reservation charge) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Routing owner is not prepared");
  owner->service->submit_artifact(make_budgeted_shared<RoutingJob>(owner->budgets.host, MemoryKind::Control,
                                                                   owner->device->source().producer, owner,
                                                                   std::move(records), std::move(charge)));
}
std::uint64_t record_host_graph(std::uint32_t kind, std::uint32_t phase, std::string_view backend,
                                std::uint64_t identity, std::uint64_t template_sequence, std::uint64_t time_ns) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner || !owner->host_channel)
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, 0, "Graph Source is not prepared");
  const auto source = owner->host->source();
  auto &channel = *owner->host_channel;
  if (channel.collector().stop_requested() || channel.buffer().progress().counts.stopped)
    return 0;
  if (owner->host->process_role() != xpool::RuntimeRole::Instance || kind < 500 || kind > 506 || phase < 1 ||
      phase > 2 || backend.empty() || backend.size() > 127 ||
      ((kind == 500 || kind == 504) ? identity != 0 : identity == 0))
    throw CollectionError(CollectionErrorCode::MetadataUnavailable, source.producer,
                          "invalid SGLang Graph boundary or identity");
  const auto lifecycle = kind <= 503;
  if (identity == 0) {
    identity = lifecycle ? owner->templates.try_next() : owner->executions.try_next();
    if (!identity || (!lifecycle && identity >= (std::uint64_t{1} << 63))) {
      channel.collector().fail(StopReason::IdentityExhausted);
      channel.buffer().stop();
      throw CollectionError(CollectionErrorCode::ResourceBudget, source.producer, "Graph identities exhausted");
    }
    if (lifecycle) {
      try {
        owner->host->add_template(source.producer, identity, phase == 1 ? "sglang_decode" : "sglang_prefill", backend);
        owner->host->ensure_revision();
      } catch (...) {
        channel.collector().fail(StopReason::MetadataUnavailable);
        channel.buffer().stop();
        throw;
      }
    }
  }
  auto event = Event{.raw_time_ns = time_ns,
                     .clock_domain_id = source.clock_domain_id,
                     .kind = kind,
                     .presence = lifecycle ? 1U << 3 : (1U << 4) | (template_sequence ? 1U << 3 : 0),
                     .graph_owner = source.producer,
                     .graph_template = lifecycle ? identity : template_sequence,
                     .graph_execution = lifecycle ? 0 : identity,
                     .role = 1,
                     .data0 = 1,
                     .data1 = phase,
                     .data4 = kind == 502   ? 1U
                              : kind == 506 ? 3U
                                            : 0U};
  auto &buffer = owner->host_channel->buffer();
  const auto reserved = try_reserve_host(buffer);
  if (reserved.status == ReserveStatus::Reserved)
    commit_host(buffer, reserved.token, event);
  return identity;
}
bool finish_attachments(bool producers_retired) {
  const auto lock = std::lock_guard{owner_mutex};
  if (!owner)
    return true;
  if (owner->borrowed) {
    if (!owner->service->joined())
      return false;
    const auto service = owner->service;
    owner.reset();
    service->rethrow_failure();
    return true;
  }
  const auto policy = ProcessPreparation::singleton().read();
  const auto wait = std::chrono::milliseconds{policy->limits.shutdown_wait_ms};
  if (!owner->service->shutdown(ProcessPreparation::singleton().begin_retirement(wait), producers_retired, wait))
    return false;
  const auto service = owner->service;
  owner.reset();
  service->rethrow_failure();
  return true;
}
} // namespace xpool::devkit::timeline
