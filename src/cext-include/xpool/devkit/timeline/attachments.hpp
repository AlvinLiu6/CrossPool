#pragma once

/// \file xpool/devkit/timeline/attachments.hpp
/// \brief Registered Graph structures and Routing tensor publication ownership.
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ATen/core/Tensor.h>
#include <cuda_runtime.h>

#include <xpool/devkit/ffn_routing_observer.hpp>
#include <xpool/devkit/timeline/metaregistry.hpp>
#include <xpool/devkit/timeline/service.hpp>

namespace xpool::devkit::timeline {
/// One immutable Routing payload and its original Device publication facts.
struct RoutingAttachment {
  /// Device-allocated attachment Attempt; independent of retained tensor position.
  std::uint64_t sequence;
  /// Registered Fabric Generation reference at publication.
  std::uint64_t generation;
  /// Original zero-based Invocation component.
  std::uint64_t instance_index;
  /// Original positive production Invocation sequence.
  std::uint64_t invocation_sequence;
  /// Publishing Executor Lane; zero is valid.
  std::uint64_t lane_index;
  /// Positive production Lane lease sequence.
  std::uint64_t lease_sequence;
  /// Original zero-based model FFN layer.
  std::uint64_t layer_ordinal;
  /// Publishing FfnAgent PE from registered Fabric membership.
  std::int32_t pe;
  /// Registered Device raw-clock domain.
  std::uint64_t clock_domain_id;
  /// Original Device publication time in nanoseconds; zero is valid.
  std::uint64_t raw_publication_time_ns;
  /// Contiguous CPU int32 tensor with live_rows by effective_topk shape.
  at::Tensor topk_ids;
  /// Contiguous CPU float32 tensor matching topk_ids dimensions.
  at::Tensor topk_weights;
};
/// Immutable Graph inventory, independent of any public getter shape.
struct GraphStructure {
  /// Borrow the process budget allocator through every queued inventory read.
  explicit GraphStructure(std::pmr::memory_resource *resource) : node_counts(resource) {}
  /// Actual CUDA node-kind inventory, with nested Graphs traversed.
  std::pmr::map<cudaGraphNodeType, std::size_t> node_counts;
  /// Actual captured model binding sites.
  std::size_t binding_site_count = 0;
  /// Installed primary-compute branches of this Lane Graph.
  std::size_t compute_branch_count = 0;
  /// Installed production delivery branches of this Lane Graph.
  std::size_t delivery_branch_count = 0;
};
/// Writer serialization bridge to the installed Safetensors library. The
/// binding installs a stateless function; no Python object crosses lifetimes.
using RoutingEncoder = std::string (*)(BatchSource, std::span<const RoutingAttachment>);
/// Install the language serialization boundary once during module initialization.
void install_routing_encoder(RoutingEncoder encoder);
/// Prepare attachment Sources on the same admitted process service. Paths come
/// from resolved configuration. Existing Fabric Device storage is borrowed.
void prepare_attachments(std::shared_ptr<MetadataRegistry> host, std::shared_ptr<MetadataRegistry> device, int ordinal,
                         std::uint64_t generation, std::string_view graph_outdir, std::string_view routing_outdir,
                         xpool::devkit::ffn_routing_observer::MemoryDemand routing_demand);
/// Construct inventory containers on the actual process Host ledger.
GraphStructure make_graph_structure();
/// Register an actual installed template and queue its immutable inventory.
void submit_graph_structure(GraphStructure structure, bool lane, std::size_t index);
/// Immutable borrowed Source and Fabric membership installed before execution.
struct RoutingSource {
  /// Shared Device Event pool; runtime owns retirement.
  BufferView events;
  /// Registered active Fabric Generation reference.
  std::uint64_t generation;
  /// Actual publishing FfnAgent PE.
  std::int32_t pe;
};
/// Borrow the original preparation facts for Routing publication.
RoutingSource routing_source();
/// Borrow the already activated Device Source for the real Routing Hook.
BufferView routing_event_buffer();
/// Queue a drained Routing payload. The retained reservation precedes all CPU
/// tensor and header allocations and survives until Writer serialization ends.
void submit_routing_attachments(std::vector<RoutingAttachment> records, MemoryBudget::Reservation charge);
/// Submit independently observed attachment outcome totals once after drain.
void record_routing_counts(std::uint64_t attempted, std::uint64_t full, std::uint64_t oversized);
/// Mark an uncertain Routing copy independently of any prior stop reason.
[[noreturn]] void fail_routing_transfer(std::string_view operation, std::string_view cause);
/// Admit independent Routing Device payload before CUDA allocation.
MemoryBudget::Reservation admit_routing_device(std::size_t bytes);
/// Admit complete Host copy/serialization coexistence before copying tensors.
MemoryBudget::Reservation admit_routing_copy(std::size_t bytes);
/// Record real Host Graph boundaries on its registered Source. The returned
/// identity is reused by subsequent boundaries of that same call or template.
std::uint64_t record_host_graph(std::uint32_t kind, std::uint32_t phase, std::string_view backend,
                                std::uint64_t identity, std::uint64_t template_sequence, std::uint64_t time_ns);
/// Join standalone Sources after the runtime proves all users retired. A shared
/// Fabric owner performs the joint shutdown instead. Never supplies Device proof.
bool finish_attachments(bool producers_retired);
} // namespace xpool::devkit::timeline
