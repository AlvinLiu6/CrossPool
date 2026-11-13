#include <algorithm>
#include <cstring>

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/digest.hpp>
#include <xpool/devkit/timeline/metastate.hpp>

namespace xpool::devkit::timeline::detail {
namespace {

void uuid(JsonOutput &out, const IdentityUuid &value) {
  const auto text = format_identity_uuid(value);
  out.quoted({text.data(), text.size()});
}
void hash(JsonOutput &out, const std::array<std::byte, 32> &value) {
  constexpr auto hex = std::string_view{"0123456789abcdef"};
  auto text = std::array<char, 64>{};
  for (auto index = std::size_t{0}; index < value.size(); ++index) {
    const auto byte = std::to_integer<unsigned>(value[index]);
    text[index * 2] = hex[byte >> 4];
    text[index * 2 + 1] = hex[byte & 15];
  }
  out.quoted({text.data(), text.size()});
}
void process(JsonOutput &out, const ProcessIdentity &value) {
  out.text("{\"creation_id\":");
  uuid(out, value.creation_id);
  out.text(",\"pid\":");
  out.number(value.pid);
  out.text(",\"runtime_role\":");
  out.quoted(xpool::name(value.runtime_role));
  out.text("}");
}
std::string_view capability(MetadataCapability value) {
  switch (value) {
  case MetadataCapability::Enabled:
    return "enabled";
  case MetadataCapability::Disabled:
    return "disabled";
  case MetadataCapability::NotApplicable:
    return "not_applicable";
  case MetadataCapability::Unavailable:
    return "unavailable";
  }
  return {};
}
std::string_view artifact_status(MetadataArtifactStatus value) {
  switch (value) {
  case MetadataArtifactStatus::Published:
    return "published";
  case MetadataArtifactStatus::Unavailable:
    return "unavailable";
  case MetadataArtifactStatus::Missing:
    return "missing";
  case MetadataArtifactStatus::PublishFailed:
    return "publish_failed";
  }
  return {};
}
void configuration(JsonOutput &out, const CollectionPolicy &policy) {
  const auto boolean = [&](bool value) { out.text(value ? "true" : "false"); };
  out.text("{\"timeline\":");
  boolean(policy.unified);
  const auto number = [&](std::string_view key, std::uint64_t value) {
    out.text(",");
    out.key(key);
    out.number(value);
  };
  const auto &limits = policy.limits;
  number("device_bytes", limits.device_bytes);
  number("host_bytes", limits.host_bytes);
  number("chunk_bytes", limits.chunk_bytes);
  number("collection_interval_ms", limits.collection_interval_ms);
  number("disk_bytes", limits.disk_bytes);
  number("metadata_reserve_bytes", limits.metadata_reserve_bytes);
  number("shutdown_wait_ms", limits.shutdown_wait_ms);
  number("calibration_block_attempts", limits.calibration_block_attempts);
  number("device_event_slot_capacity", limits.device_event_slot_capacity);
  number("endpoint_capacity", limits.endpoint_capacity);
  number("context_slot_capacity", limits.context_slot_capacity);
  number("sampler_slot_capacity", limits.sampler_slot_capacity);
  const auto observer = [&](std::string_view key, const xpool::debug::TraceObserverOptions &value) {
    out.text(",");
    out.key(key);
    out.text("{\"enable\":");
    boolean(value.enable);
    out.text(",\"record_capacity\":");
    out.number(value.record_capacity);
    out.text("}");
  };
  observer("transport_observer", policy.legacy.transport_observer);
  observer("fabric_observer", policy.legacy.fabric_observer);
  out.text(",\"graph_observer\":{\"enable\":");
  boolean(policy.legacy.graph_observer.enable);
  out.text("}");
  out.text(",\"ffn_routing_observer\":{\"enable\":");
  boolean(policy.legacy.ffn_routing_observer.enable);
  out.text(",\"record_capacity\":");
  out.number(policy.legacy.ffn_routing_observer.record_capacity);
  out.text("}");
  out.text(",\"explicit_transport_identity_consumer\":");
  boolean(policy.explicit_transport_identity_consumer);
  out.text(",\"debug_configuration_sha256\":");
  hash(out, policy.configuration_fingerprint);
  out.text(",\"unified_outdir_sha256\":");
  if (policy.unified_outdir_fingerprint)
    hash(out, *policy.unified_outdir_fingerprint);
  else
    out.text("null");
  out.text("}");
}

} // namespace

std::span<const std::byte> MetadataState::render(std::uint64_t revision) {
  if (revision == 0)
    invalid("xpool metadata revision must be positive");
  validate();
  rendered_revision = 0;
  document_size = 0;
  auto out = JsonOutput{{this, [](void *context, std::string_view bytes) {
                           auto &state = *static_cast<MetadataState *>(context);
                           state.capacity(bytes.size() <= state.limits.document_bytes - state.document_size);
                           std::memcpy(state.document + state.document_size, bytes.data(), bytes.size());
                           state.document_size += bytes.size();
                         }}};
  const auto *own = producer(source.producer);
  out.text("{\"contract_version\":");
  out.number(kContractVersion);
  out.text(",\"run_id\":");
  uuid(out, source.run_id);
  out.text(",\"producer_id\":");
  out.number(source.producer);
  out.text(",\"revision\":");
  out.number(revision);
  out.text(",\"process\":");
  process(out, own->process.process);
  out.text(",\"clock_domain_id\":");
  out.number(source.clock_domain_id);
  out.text(",\"producers\":[");
  auto comma = false;
  for (const auto &entry : producers) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"id\":");
    out.number(entry.source.producer);
    out.text(",\"source\":");
    out.quoted(entry.device ? "device" : "host");
    out.text(",\"process\":");
    process(out, entry.process.process);
    out.text(",\"process_creation_id\":");
    uuid(out, entry.process.process.creation_id);
    out.text(",\"clock_domain_id\":");
    out.number(entry.source.clock_domain_id);
    out.text("}");
  }
  out.text("],\"generations\":[");
  comma = false;
  for (const auto &entry : generations) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"id\":");
    out.number(entry.id);
    out.text(",\"fabric_generation\":");
    out.quoted(entry.production);
    out.text(",\"atnagent_count\":");
    out.number(entry.atn_count);
    out.text(",\"ffnagent_count\":");
    out.number(entry.ffn_count);
    out.text(",\"model_topologies\":[");
    auto model_comma = false;
    for (const auto &model : entry.models) {
      if (model_comma)
        out.text(",");
      model_comma = true;
      out.text("{\"atn_tp_size\":");
      out.number(model.tp);
      out.text(",\"atn_dp_size\":");
      out.number(model.dp);
      if (model.ffn_tp) {
        out.text(",\"ffn_tp_size\":");
        out.number(*model.ffn_tp);
      }
      out.text("}");
    }
    out.text("],\"pe_members\":[");
    auto member_comma = false;
    for (const auto &member : entry.pe_members) {
      if (member_comma)
        out.text(",");
      member_comma = true;
      out.text("{\"pe\":");
      out.number(static_cast<std::uint64_t>(member.pe));
      out.text(",\"host_producer\":");
      out.number(member.host_producer);
      out.text(",\"device_producer\":");
      out.number(member.device_producer);
      out.text("}");
    }
    out.text("],\"scheduler\":");
    out.quoted(entry.fifo ? "fifo" : "random");
    out.text(",\"models\":[");
    model_comma = false;
    for (const auto &model : entry.models) {
      if (model_comma)
        out.text(",");
      model_comma = true;
      out.text("{\"instance_index\":");
      out.number(model.instance_index);
      out.text(",\"model_id\":");
      out.quoted(model.model_id);
      out.text("}");
    }
    out.text("]}");
  }
  out.text("],\"clock_domains\":[");
  comma = false;
  for (const auto &entry : clocks) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"id\":");
    out.number(entry.id);
    out.text(",\"kind\":");
    out.quoted(entry.device ? "device_globaltimer" : "host_monotonic");
    out.text(",\"unit\":\"ns\"");
    if (entry.device) {
      out.text(",\"process_creation_id\":");
      uuid(out, entry.owner);
      out.text(",\"physical_device_uuid\":");
      out.quoted(entry.physical);
      out.text(",\"logical_device\":");
      out.number(entry.logical_device);
      out.text(",\"raw_epoch\":");
      out.number(entry.epoch);
    } else {
      out.text(",\"host_creation_id\":");
      uuid(out, entry.owner);
    }
    out.text("}");
  }
  out.text("],\"endpoints\":[");
  comma = false;
  for (const auto &entry : endpoints) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"owner_producer\":");
    out.number(entry.owner);
    out.text(",\"endpoint_sequence\":");
    out.number(entry.sequence);
    out.text(",\"generation\":");
    out.number(entry.generation);
    out.text(",\"instance_index\":");
    out.number(entry.instance_index);
    out.text(",\"instance_rank\":");
    out.number(entry.rank);
    out.text(",\"model_id\":");
    out.quoted(entry.model_id);
    out.text(",\"arena_handle_suffix\":");
    out.quoted({entry.suffix.data(), entry.suffix.size()});
    out.text(",\"sites\":[{\"producer\":");
    out.number(entry.instance_producer);
    out.text(",\"site\":\"instance\"},{\"producer\":");
    out.number(entry.owner);
    out.text(",\"site\":\"atnagent\"}]");
    if (entry.supersedes) {
      out.text(",\"supersedes_endpoint_sequence\":");
      out.number(*entry.supersedes);
    }
    out.text("}");
  }
  out.text("],\"graph_templates\":[");
  comma = false;
  for (const auto &entry : templates) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"owner_producer\":");
    out.number(entry.owner);
    out.text(",\"template_sequence\":");
    out.number(entry.sequence);
    out.text(",\"kind\":");
    out.quoted(entry.kind);
    if (entry.has_backend) {
      out.text(",\"backend_class\":");
      out.quoted(entry.backend);
    }
    if (entry.signature) {
      out.text(",\"signature_index\":");
      out.number(*entry.signature);
    }
    if (entry.lane) {
      out.text(",\"lane_index\":");
      out.number(*entry.lane);
    }
    out.text(",\"compute_templates\":[");
    auto child_comma = false;
    for (const auto child : entry.compute) {
      if (child_comma)
        out.text(",");
      child_comma = true;
      out.number(child);
    }
    out.text("]}");
  }
  out.text("],\"artifacts\":[");
  comma = false;
  for (const auto &entry : artifacts) {
    if (comma)
      out.text(",");
    comma = true;
    out.text("{\"id\":");
    out.number(entry.id);
    out.text(",\"kind\":");
    out.quoted(entry.kind);
    out.text(",\"status\":");
    out.quoted(artifact_status(entry.status));
    out.text(",\"path\":");
    if (entry.has_path)
      out.quoted(entry.path);
    else
      out.text("null");
    out.text(",\"sha256\":");
    if (entry.sha256)
      hash(out, *entry.sha256);
    else
      out.text("null");
    out.text("}");
  }
  out.text("],\"capabilities\":{\"transport\":");
  out.quoted(capability(capabilities.transport));
  out.text(",\"fabric\":");
  out.quoted(capability(capabilities.fabric));
  out.text(",\"graph\":");
  out.quoted(capability(capabilities.graph));
  out.text(",\"routing\":");
  out.quoted(capability(capabilities.routing));
  out.text(",\"calibration\":");
  out.quoted(capability(capabilities.calibration));
  out.text("},\"configuration\":");
  configuration(out, policy);
  out.text("}\n");
  const auto total = out.finish();
  if (total != document_size)
    invalid("xpool metadata serialization size disagrees with its bounded sink");
  const auto bytes = std::span<const std::byte>{document, document_size};
  rendered_hash = sha256(bytes);
  rendered_revision = revision;
  return bytes;
}

} // namespace xpool::devkit::timeline::detail
