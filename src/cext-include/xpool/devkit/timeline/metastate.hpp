#pragma once

/// \file xpool/devkit/timeline/metastate.hpp
/// \brief Internal semantic state retained by metadata coverage callbacks.

#include <mutex>
#include <vector>

#include <xpool/devkit/timeline/json.hpp>
#include <xpool/devkit/timeline/metaregistry.hpp>

/// \cond INTERNAL
namespace xpool::devkit::timeline::detail {

struct MetadataProducer {
  RegisteredProcess process;
  BatchSource source;
  bool device;
  std::uint64_t since = 0;
};
struct MetadataClock {
  std::uint64_t id;
  bool device;
  IdentityUuid owner;
  std::uint32_t logical_device;
  std::uint64_t epoch;
  std::pmr::string physical;
  std::uint64_t since = 0;
};
struct MetadataModel {
  std::uint64_t instance_index;
  std::pmr::string model_id;
  std::uint64_t tp;
  std::uint64_t dp;
  std::optional<std::uint64_t> ffn_tp;
};
struct MetadataGeneration {
  std::uint64_t id;
  std::pmr::string production;
  std::uint64_t atn_count;
  std::uint64_t ffn_count;
  bool fifo;
  std::pmr::vector<MetadataModel> models;
  std::pmr::vector<MetadataPeMember> pe_members;
  std::uint64_t since = 0;
};
struct MetadataEndpoint {
  std::uint64_t owner;
  std::uint64_t sequence;
  std::uint64_t generation;
  std::uint64_t instance_index;
  std::uint32_t rank;
  std::pmr::string model_id;
  std::array<char, 16> suffix;
  std::uint64_t instance_producer;
  std::optional<std::uint64_t> supersedes;
  std::uint64_t since = 0;
};
struct MetadataTemplate {
  std::uint64_t owner;
  std::uint64_t sequence;
  std::pmr::string kind;
  std::pmr::string backend;
  bool has_backend;
  std::optional<std::uint64_t> signature;
  std::optional<std::uint64_t> lane;
  std::pmr::vector<std::uint64_t> compute;
  std::uint64_t since = 0;
};
struct MetadataArtifact {
  std::uint64_t id;
  std::pmr::string kind;
  MetadataArtifactStatus status;
  std::pmr::string path;
  bool has_path;
  std::optional<std::array<std::byte, 32>> sha256;
  std::uint64_t since = 0;
};

// The catalog retains this state for coverage after the initiating registry
// handle retires. This state only weakly references its byte catalog.
struct MetadataState {
  MetadataState(std::shared_ptr<MemoryBudget> host, BatchSource source, CollectionPolicy policy,
                MetadataCapabilities capabilities, MetadataLimits limits);
  ~MetadataState();
  void add_host(RegisteredProcess process);
  void add_clock(std::uint64_t id, bool device, IdentityUuid owner, std::uint32_t logical_device, std::uint64_t epoch,
                 std::string_view physical);
  void validate() const;
  void validate_events(std::span<const Event> events, std::uint64_t revision) const;
  std::span<const std::byte> render(std::uint64_t revision);
  void mark_registered(std::uint64_t revision);
  MetadataRevision append(bool only_if_changed);
  void changed() noexcept { rendered_revision = 0; }
  [[noreturn]] void invalid(std::string_view message) const;
  [[noreturn]] void invalid_membership(std::string_view code, std::uint64_t generation, std::optional<std::int32_t> pe,
                                       std::uint64_t producer, std::string_view detail) const;
  void capacity(bool available) const;
  const MetadataProducer *producer(std::uint64_t id) const;
  const MetadataClock *clock(std::uint64_t id) const;
  const MetadataGeneration *generation(std::uint64_t id) const;
  const MetadataEndpoint *endpoint(std::uint64_t owner, std::uint64_t sequence) const;
  const MetadataTemplate *graph_template(std::uint64_t owner, std::uint64_t sequence) const;
  const MetadataArtifact *artifact(std::uint64_t id) const;

  const std::shared_ptr<MemoryBudget> host;
  const BatchSource source;
  const CollectionPolicy policy;
  const MetadataCapabilities capabilities;
  const MetadataLimits limits;
  BudgetMemoryResource resource;
  std::pmr::vector<MetadataProducer> producers;
  std::pmr::vector<MetadataClock> clocks;
  std::pmr::vector<MetadataGeneration> generations;
  std::pmr::vector<MetadataEndpoint> endpoints;
  std::pmr::vector<MetadataTemplate> templates;
  std::pmr::vector<MetadataArtifact> artifacts;
  std::weak_ptr<MetadataCatalog> catalog;
  bool catalog_bound = false;
  std::byte *document = nullptr;
  std::size_t document_size = 0;
  std::uint64_t rendered_revision = 0;
  std::array<std::byte, 32> rendered_hash{};
  std::mutex mutex;
};

} // namespace xpool::devkit::timeline::detail
/// \endcond
