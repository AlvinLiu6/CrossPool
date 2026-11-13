#include <xpool/devkit/timeline/error.hpp>
#include <xpool/devkit/timeline/metastate.hpp>

namespace xpool::devkit::timeline::detail {
namespace {
bool valid_capability(MetadataCapability value) {
  return value == MetadataCapability::Enabled || value == MetadataCapability::Disabled ||
         value == MetadataCapability::NotApplicable || value == MetadataCapability::Unavailable;
}
bool valid_status(MetadataArtifactStatus value) {
  return value == MetadataArtifactStatus::Published || value == MetadataArtifactStatus::Unavailable ||
         value == MetadataArtifactStatus::Missing || value == MetadataArtifactStatus::PublishFailed;
}
template <typename T> bool registered(const T *value, std::uint64_t revision) {
  return value && value->since != 0 && value->since <= revision;
}
std::uint32_t event_role(xpool::RuntimeRole role) {
  switch (role) {
  case xpool::RuntimeRole::Instance:
    return 1;
  case xpool::RuntimeRole::AtnAgent:
    return 2;
  case xpool::RuntimeRole::FfnAgent:
    return 4;
  case xpool::RuntimeRole::Daemon:
    return 5;
  }
  return 0;
}
} // namespace

void MetadataState::validate() const {
  const auto *own = producer(source.producer);
  if (!own || own->source.clock_domain_id != source.clock_domain_id || own->source.run_id != source.run_id)
    invalid("xpool metadata has no matching complete owner Producer");
  for (const auto &entry : producers) {
    const auto *domain = clock(entry.source.clock_domain_id);
    if (!domain || domain->device != entry.device ||
        domain->owner != (entry.device ? entry.process.process.creation_id : entry.process.process.host_creation_id))
      invalid("xpool metadata Producer has no matching actual clock domain");
  }
  for (const auto &entry : generations) {
    for (auto index = std::size_t{0}; index < entry.pe_members.size(); ++index) {
      const auto &member = entry.pe_members[index];
      if (member.host_producer == member.device_producer)
        invalid_membership("identity_conflict", entry.id, member.pe, member.host_producer, "repeated Producer binding");
      for (auto prior = std::size_t{0}; prior < index; ++prior) {
        const auto &other = entry.pe_members[prior];
        if (member.pe == other.pe || member.host_producer == other.host_producer ||
            member.device_producer == other.device_producer || member.host_producer == other.device_producer ||
            member.device_producer == other.host_producer)
          invalid_membership("identity_conflict", entry.id, member.pe, member.host_producer,
                             "duplicate PE or Producer binding");
      }
    }
    constexpr auto maximum_members = static_cast<std::uint64_t>(INT32_MAX) + 1;
    if (entry.atn_count > maximum_members || entry.ffn_count > maximum_members - entry.atn_count ||
        entry.pe_members.size() != entry.atn_count + entry.ffn_count)
      invalid_membership("invalid_event", entry.id, std::nullopt, 0, "missing or out-of-range PE coverage");
    for (auto index = std::size_t{0}; index < entry.pe_members.size(); ++index) {
      const auto &member = entry.pe_members[index];
      if (member.pe < 0 || static_cast<std::uint64_t>(member.pe) != index)
        invalid_membership("invalid_event", entry.id, member.pe, member.host_producer, "PE order or coverage gap");
      const auto *host = producer(member.host_producer);
      const auto *device = producer(member.device_producer);
      if (!host || host->device || !device || !device->device || host->process.process != device->process.process ||
          device->process.host.producer != member.host_producer)
        invalid_membership("invalid_event", entry.id, member.pe, member.host_producer,
                           "unresolved or mismatched Host/Device owners");
      const auto expected = index < entry.atn_count ? xpool::RuntimeRole::AtnAgent : xpool::RuntimeRole::FfnAgent;
      if (host->process.process.runtime_role != expected)
        invalid_membership("invalid_event", entry.id, member.pe, member.host_producer, "wrong process role placement");
    }
    if (entry.models.empty())
      invalid("xpool metadata Generation has no complete model topology");
  }
  for (const auto &entry : endpoints) {
    const auto *owner = producer(entry.owner);
    const auto *instance = producer(entry.instance_producer);
    const auto *gen = generation(entry.generation);
    if (!owner || !owner->device || owner->process.process.runtime_role != xpool::RuntimeRole::AtnAgent || !instance ||
        !instance->device || instance->process.process.runtime_role != xpool::RuntimeRole::Instance || !gen ||
        entry.instance_index >= gen->models.size() ||
        gen->models[static_cast<std::size_t>(entry.instance_index)].model_id != entry.model_id)
      invalid("xpool metadata Endpoint has unresolved owner, site, Generation or model facts");
    if (entry.supersedes && (!endpoint(entry.owner, *entry.supersedes) || *entry.supersedes >= entry.sequence))
      invalid("xpool metadata Endpoint supersession must reference its prior owner-local lifetime");
  }
  for (const auto &entry : templates) {
    const auto *owner = producer(entry.owner);
    if (!owner || owner->device)
      invalid("xpool metadata Graph template owner must be a registered Host Producer");
    for (const auto child : entry.compute)
      if (!graph_template(entry.owner, child))
        invalid("xpool metadata compute template has no matching same-owner definition");
  }
  for (const auto &entry : artifacts)
    if (!valid_status(entry.status))
      invalid("xpool metadata artifact outcome is outside the C0 vocabulary");
  if (policy.unified &&
      (capabilities.transport != MetadataCapability::Enabled || capabilities.fabric != MetadataCapability::Enabled ||
       capabilities.graph != MetadataCapability::Enabled || capabilities.calibration != MetadataCapability::Enabled ||
       (capabilities.routing != MetadataCapability::Enabled &&
        capabilities.routing != MetadataCapability::NotApplicable)))
    invalid("xpool unified metadata requires every applicable provider and calibration");
  for (const auto value : {capabilities.transport, capabilities.fabric, capabilities.graph, capabilities.routing,
                           capabilities.calibration})
    if (!valid_capability(value))
      invalid("xpool metadata capability is outside the C0 vocabulary");
}

void MetadataState::validate_events(std::span<const Event> events, std::uint64_t revision) const {
  try {
    for (const auto &event : events)
      event.validate(source.producer, source.clock_domain_id);
  } catch (const ParseError &error) {
    invalid(error.what());
  }
  const auto *own = producer(source.producer);
  if (!registered(own, revision) || !registered(clock(source.clock_domain_id), revision))
    invalid("xpool Event owner/clock is absent from its selected metadata revision");
  for (const auto &event : events) {
    const auto enabled = MetadataCapability::Enabled;
    if ((((event.kind >= 100 && event.kind <= 110) || event.kind == 150) && capabilities.transport != enabled) ||
        (event.kind >= 200 && event.kind <= 408 && capabilities.fabric != enabled) ||
        (event.kind == 450 && (capabilities.transport != enabled || capabilities.fabric != enabled)) ||
        (event.kind >= 500 && event.kind <= 512 && capabilities.graph != enabled) ||
        (event.kind == 550 && capabilities.routing != enabled) ||
        ((event.kind == 600 || event.kind == 601) && capabilities.calibration != enabled))
      invalid("xpool Event family is not enabled in its immutable capabilities");
    const auto present = [&](unsigned bit) { return (event.presence & (std::uint32_t{1} << bit)) != 0; };
    const auto require = [&](bool valid, std::string_view message) {
      if (!valid)
        invalid(message);
    };
    const auto graph_device = event.kind == 507 || event.kind == 508;
    const auto graph_host = event.kind >= 500 && event.kind <= 512 && !graph_device;
    const auto sample = event.kind == 600 || event.kind == 601;
    require(own->device != (graph_host || sample), "xpool Event family has the wrong Host/Device Source");
    const MetadataGeneration *gen = nullptr;
    if (present(0)) {
      gen = generation(event.generation);
      require(registered(gen, revision), "xpool Event Generation is absent from its selected metadata revision");
    }
    const MetadataPeMember *member = nullptr;
    if (gen)
      for (const auto &candidate : gen->pe_members)
        if (candidate.host_producer == source.producer || candidate.device_producer == source.producer)
          member = &candidate;
    if (present(17)) {
      if (!gen || !member)
        invalid_membership("invalid_event", event.generation, event.pe, source.producer,
                           "Event PE requires its own Generation membership");
      if (event.pe != member->pe)
        invalid_membership("identity_conflict", event.generation, event.pe, source.producer,
                           "Event PE disagrees with membership");
    }
    const auto expected_role = event_role(own->process.process.runtime_role);
    if (sample)
      require(event.role == 5, "xpool calibration Event must retain collector observation role");
    else
      require(event.role == expected_role || (expected_role == 4 && event.role == 3 && gen && member &&
                                              static_cast<std::uint64_t>(member->pe) == gen->atn_count),
              "xpool Event role disagrees with its process or Coordinator first-FfnAgent membership");
    if (present(1)) {
      const auto *ep = endpoint(event.endpoint_owner, event.endpoint_sequence);
      require(registered(ep, revision), "xpool Event Endpoint is absent from its selected metadata revision");
      if (gen)
        require(ep->generation == gen->id, "xpool Event Endpoint and Generation facts disagree");
      if ((event.kind >= 100 && event.kind <= 110) || event.kind == 150 || event.kind == 450) {
        require(source.producer == (event.role == 1 ? ep->instance_producer : ep->owner),
                "xpool Transport Event has the wrong registered Endpoint site");
        require(event.instance_rank == ep->rank, "xpool Transport Event rank disagrees with Endpoint metadata");
        if (present(5))
          require(event.instance_index == ep->instance_index,
                  "xpool associated Invocation and Endpoint Instance facts disagree");
      }
    }
    if (present(5))
      require(gen && event.instance_index < gen->models.size(), "xpool Invocation has no registered Instance topology");
    if (present(3) || present(4)) {
      const auto *owner = producer(event.graph_owner);
      require(registered(owner, revision) && !owner->device &&
                  owner->process.process.creation_id == own->process.process.creation_id,
              "xpool Graph identity must name the actual owning process Host Producer");
      if (present(3))
        require(registered(graph_template(event.graph_owner, event.graph_template), revision),
                "xpool Graph template is absent from its selected metadata revision");
    }
    if (present(7)) {
      const auto *owner = producer(event.attachment_owner);
      require(registered(owner, revision) && owner->device &&
                  owner->process.process.runtime_role == xpool::RuntimeRole::FfnAgent,
              "xpool Routing attachment owner has no registered FFN Device Producer");
    }
    if (event.kind >= 100 && event.kind <= 110) {
      const auto instance_kind = event.kind <= 102 || event.kind >= 107;
      require(instance_kind == (event.role == 1), "xpool Transport boundary belongs to the other Endpoint site");
    }
    if (event.kind >= 300 && event.kind <= 307)
      require(gen && present(13) == gen->fifo, "xpool Coordinator ticket presence disagrees with its actual scheduler");
    if (((event.kind >= 500 && event.kind <= 512) || event.kind == 550) && event.data2 != 0) {
      const auto *entry = artifact(event.data2);
      require(registered(entry, revision), "xpool Event artifact is absent from its selected metadata revision");
      if (event.kind == 550)
        require(entry->status == MetadataArtifactStatus::Published && entry->kind == "routing" && entry->has_path &&
                    !entry->path.empty() && entry->sha256,
                "xpool Routing Event artifact reference requires actual publication");
    }
    if (sample) {
      const auto *device = clock(event.data3);
      require(registered(device, revision) && device->device && device->owner == own->process.process.creation_id,
              "xpool calibration Event has no same-process Device clock definition");
    }
  }
}

} // namespace xpool::devkit::timeline::detail
