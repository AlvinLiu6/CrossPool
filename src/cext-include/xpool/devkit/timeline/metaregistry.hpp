#pragma once

/// \file xpool/devkit/timeline/metaregistry.hpp
/// \brief Bounded typed C0 metadata definitions, serialization and coverage.

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

#include <xpool/devkit/timeline/identity.hpp>
#include <xpool/devkit/timeline/metadata.hpp>
#include <xpool/devkit/timeline/policy.hpp>
#include <xpool/fabric/projection.hpp>

namespace xpool::devkit::timeline {

/// C0 capability declaration. NotApplicable requires actual production facts;
/// an observer's disabled flag alone never proves a branch inapplicable.
enum class MetadataCapability { Enabled, Disabled, NotApplicable, Unavailable };
/// Immutable process policy/capability projection for the metadata owner.
struct MetadataCapabilities {
  /// Transport capture capability.
  MetadataCapability transport = MetadataCapability::Unavailable;
  /// Fabric capture capability.
  MetadataCapability fabric = MetadataCapability::Unavailable;
  /// Graph capture capability.
  MetadataCapability graph = MetadataCapability::Unavailable;
  /// Routing capability or actual lifetime-wide inapplicability proof.
  MetadataCapability routing = MetadataCapability::Unavailable;
  /// Clock calibration capability.
  MetadataCapability calibration = MetadataCapability::Unavailable;
  /// Compare the immutable capability declaration.
  bool operator==(const MetadataCapabilities &) const = default;
};

/// Fixed metadata registry geometry, admitted before related Event production.
struct MetadataLimits {
  /// Host plus Device Producer definitions in one complete document.
  std::size_t producers = 128;
  /// Raw clock domains, including referenced remote owners.
  std::size_t clocks = 128;
  /// Retained immutable production Generation definitions.
  std::size_t generations = 64;
  /// Actual model entries/topologies in each Generation.
  std::size_t models_per_generation = 8;
  /// Referenced Transport Endpoint lifetimes.
  std::size_t endpoints = 64;
  /// Referenced Graph templates across their retained lifetimes.
  std::size_t templates = 256;
  /// Compute-template references in each lane template.
  std::size_t compute_templates = 64;
  /// Immutable artifact definitions, scoped to this Producer metadata.
  std::size_t artifacts = 64;
  /// Complete UTF-8 document including its final newline.
  std::size_t document_bytes = 65536;
};

/// Final immutable metadata artifact outcome; live attachment outcomes belong
/// to the separate attachment index, rather than mutation of a committed Event.
enum class MetadataArtifactStatus { Published, Unavailable, Missing, PublishFailed };

/// One Generation's actual Fabric owner; identities stay separate from PE.
struct MetadataPeMember {
  /// Zero-based authoritative Fabric Plan PE, serialized as i32.
  std::int32_t pe = -1;
  /// Observed owning process's registered Host Producer.
  std::uint64_t host_producer = 0;
  /// That same process's registered Device Producer and raw clock owner.
  std::uint64_t device_producer = 0;
  /// Compare the complete immutable binding.
  bool operator==(const MetadataPeMember &) const = default;
};

/// \cond INTERNAL
namespace detail {
struct MetadataState;
}
/// \endcond

/// Source-bound semantic registry over trusted actual preparation facts.
/// Definition identities are append-only and retain immutable prior values.
/// Serialization uses preadmitted fixed scratch and C0's exact root keys.
/// Producer/clock/Generation/topology/reference checks precede catalog append.
/// Runtime lifecycle calls serialize on the owning preparation thread; coverage
/// validation may run concurrently on the collection lane under the state lock.
class MetadataRegistry {
  class ConstructionKey {
    friend class MetadataRegistry;
    ConstructionKey() = default;
  };

public:
  /// Allocate the typed tables, complete document scratch and shared controls
  /// under the same process Host budget. Configuration is a deterministic
  /// numeric/flag/fingerprint projection of the installed xpool.config policy.
  /// \throws CollectionError on invalid geometry or admission/allocation failure.
  static std::shared_ptr<MetadataRegistry> create(std::shared_ptr<MemoryBudget> host, BatchSource source,
                                                  CollectionPolicy policy, MetadataCapabilities capabilities,
                                                  MetadataLimits limits = {});
  MetadataRegistry(const MetadataRegistry &) = delete;
  MetadataRegistry &operator=(const MetadataRegistry &) = delete;
  /// Import one exact registered Host Source and actual machine-creation clock.
  /// Identical repetitions are idempotent; changed definitions reject.
  /// \throws CollectionError on invalid facts, conflict, capacity or admission.
  void add_host(RegisteredProcess process);
  /// Import actual Device clock/mapping facts and its previously registered
  /// process Host Source. These references are assigned by the launcher.
  /// \throws CollectionError on invalid facts, conflict, capacity or admission.
  void add_device(RegisteredProcess process, BatchSource source, std::uint32_t logical_device,
                  std::string_view physical_device_uuid, std::uint64_t raw_epoch);
  /// Stage immutable Generation scalar facts before its model entries. Models
  /// must completely describe its zero-based topology before serialization.
  /// \throws CollectionError on invalid facts, conflict or exhausted capacity.
  void add_generation(std::uint64_t id, std::string_view fabric_generation, std::uint64_t atnagent_count,
                      std::uint64_t ffnagent_count, bool fifo);
  /// Stage one authoritative PE owner before publishing this Generation.
  /// Complete documents require ascending gap-free coverage with same-process
  /// Host/Device pairs and actual AtnAgent-prefix/FfnAgent-suffix placement.
  /// Repetition is idempotent; a published table cannot gain or change members.
  /// Member storage is bounded by half the configured Producer table capacity.
  /// \throws CollectionError for identity conflict, invalid facts or capacity.
  void add_pe_member(std::uint64_t generation, std::int32_t pe, std::uint64_t host_producer,
                     std::uint64_t device_producer);
  /// Resolve this Generation's registered Host or Device Producer to its PE.
  /// An Instance/nonmember returns null; missing Generation/Producer is invalid.
  /// Definitions must form a complete validated table, including on empty traces.
  /// \throws CollectionError for missing or malformed metadata.
  std::optional<std::int32_t> fabric_pe(std::uint64_t generation, std::uint64_t producer) const;
  /// Add a contiguous model/topology entry before that Generation's first
  /// catalog registration. A published Generation can only repeat equal facts.
  /// \throws CollectionError on invalid facts, conflict or exhausted capacity.
  void add_model(std::uint64_t generation, std::uint64_t instance_index, std::string_view model_id,
                 std::uint64_t atn_tp_size, std::uint64_t atn_dp_size,
                 std::optional<std::uint64_t> ffn_tp_size = std::nullopt);
  /// Register the exact paired Device Producers and metadata-only Arena suffix;
  /// the opaque IPC handle never enters metadata. A superseded Endpoint stays
  /// immutable and present for late Events.
  /// \throws CollectionError on invalid facts, conflict or exhausted capacity.
  void add_endpoint(std::uint64_t owner, std::uint64_t sequence, std::uint64_t generation, std::uint64_t instance_index,
                    std::uint32_t instance_rank, std::string_view model_id, std::string_view arena_suffix,
                    std::uint64_t instance_producer, std::optional<std::uint64_t> supersedes = std::nullopt);
  /// Register one actual Graph-owning Host Producer's immutable template.
  /// The optional fields describe observed installation facts, not handles.
  /// \throws CollectionError on invalid facts, conflict, capacity or admission.
  void add_template(std::uint64_t owner, std::uint64_t sequence, std::string_view kind,
                    std::optional<std::string_view> backend_class = std::nullopt,
                    std::optional<std::uint64_t> signature_index = std::nullopt,
                    std::optional<std::uint64_t> lane_index = std::nullopt);
  /// Add a same-owner compute-template reference before parent publication.
  /// \throws CollectionError for conflict, missing parent or capacity exhaustion.
  void add_compute_template(std::uint64_t owner, std::uint64_t parent, std::uint64_t child);
  /// Register a final Producer-scoped artifact declaration; repetition must
  /// preserve every field. Paths are relative to the Run and cannot escape it
  /// or contain NUL. A published outcome requires its actual path/hash.
  /// \throws CollectionError on invalid facts, conflict, capacity or admission.
  void add_artifact(std::uint64_t id, std::string_view kind, MetadataArtifactStatus status,
                    std::optional<std::string_view> path = std::nullopt,
                    std::optional<std::array<std::byte, 32>> sha256 = std::nullopt);
  /// Serialize a complete validated document. The returned fixed scratch view
  /// expires at the next mutation/serialization; the caller retains this owner
  /// and serializes its cold lifecycle through consumption. Performs no I/O.
  /// \throws CollectionError for incomplete references or document capacity.
  std::span<const std::byte> document(std::uint64_t revision);
  /// Attach the exact first document already copied into a channel's catalog.
  /// The catalog retains semantic validation state; that state weakly borrows
  /// the catalog, so readers cannot create a lifetime cycle.
  /// \throws CollectionError for source/hash/revision mismatch or replacement.
  void bind_catalog(std::shared_ptr<MetadataCatalog> catalog);
  /// Append the next complete validated revision before allowing new Event
  /// references. This registers bytes, not successful file publication. The
  /// independent Writer publishes the referencing revision before its chunks.
  /// \throws CollectionError for missing catalog, lifecycle, schema or capacity.
  MetadataRevision append_revision();
  /// Publish draft definitions into one complete immutable catalog revision,
  /// or reuse the current one when every definition is already registered.
  /// Cold startup/publication retries do not consume another revision slot.
  /// This registers bytes; the Writer still owns actual file publication.
  /// \throws CollectionError for incomplete drafts, missing catalog or budget.
  MetadataRevision ensure_revision();
  /// Exact immutable source registration.
  const BatchSource &source() const noexcept;
  /// Check an unbound registry's actual channel kind and Device mapping before
  /// Source admission. Null selects Host; a Device index must match its raw
  /// clock definition. Performs no CUDA or I/O and leaves definitions intact.
  /// \throws CollectionError for missing/conflicting facts or prior binding.
  void validate_preparation(std::optional<int> device_index) const;
  /// Require this Device Source's actual PE, Generation identity, scheduler and
  /// model topology to match the projection used for the real Fabric join.
  /// Fabric capability must be enabled before this owner can install a view.
  void require_fabric(std::uint64_t generation, const xpool::fabric::ArenaProjection &projection, int pe) const;
  /// Allocator-only constructor; use create() for complete shared admission.
  MetadataRegistry(ConstructionKey, std::shared_ptr<detail::MetadataState> state);

private:
  // Catalog callbacks retain the state after an initiating registry handle
  // retires; the outer handle is not required on the collection lane.
  const std::shared_ptr<detail::MetadataState> state_;
};

} // namespace xpool::devkit::timeline
