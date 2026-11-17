#include <xpool/devkit/graph_observer.hpp>

#include <algorithm>
#include <mutex>
#include <optional>
#include <ranges>

#include <c10/util/Exception.h>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/adapters.hpp>
#include <xpool/devkit/timeline/attachments.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace xpool::devkit::graph_observer {

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, HostAdapter::observe, context) {}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnPrimaryGraphParameterizePostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  auto primary_graph = timeline::make_graph_structure();
  xpool::utils::cuda::graph::for_each_reachable_node(
      context.graph, [&primary_graph](cudaGraphNode_t, cudaGraphNodeType type) { ++primary_graph.node_counts[type]; });
  primary_graph.binding_site_count = context.binding_site_count;
  timeline::submit_graph_structure(std::move(primary_graph), false, context.execution_signature_index);
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnLaneGraphBuildPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  auto lane_graph = timeline::make_graph_structure();
  xpool::utils::cuda::graph::for_each_reachable_node(
      context.graph, [&lane_graph](cudaGraphNode_t, cudaGraphNodeType type) { ++lane_graph.node_counts[type]; });
  const auto compute_branches = xpool::utils::cuda::graph::conditional_bodies(context.compute_switch);
  lane_graph.compute_branch_count = static_cast<std::size_t>(std::ranges::distance(
      compute_branches | std::views::filter([](cudaGraph_t branch) {
        return std::ranges::any_of(xpool::utils::cuda::graph::nodes(branch), [](cudaGraphNode_t node) {
          return xpool::utils::cuda::graph::node_type(node) == cudaGraphNodeTypeGraph;
        });
      })));
  const auto delivery_branches = xpool::utils::cuda::graph::conditional_bodies(context.delivery_switch);
  lane_graph.delivery_branch_count = static_cast<std::size_t>(std::ranges::distance(
      delivery_branches | std::views::filter([](cudaGraph_t branch) {
        return std::ranges::any_of(xpool::utils::cuda::graph::nodes(branch), [](cudaGraphNode_t node) {
          return xpool::utils::cuda::graph::node_type(node) == cudaGraphNodeTypeKernel;
        });
      })));

  timeline::submit_graph_structure(std::move(lane_graph), true, context.executor_lane_index);
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, HostAdapter::observe, context) {}

} // namespace xpool::devkit::graph_observer
