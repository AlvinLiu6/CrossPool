#include <cstddef>
#include <cstdint>
#include <map>

#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <xpool/bindings.hpp>
#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/ffn_routing_observer.hpp>
#include <xpool/devkit/graph_observer.hpp>
#include <xpool/devkit/timeline/attachments.hpp>
#include <xpool/devkit/timeline/fabricruntime.hpp>
#include <xpool/devkit/timeline/identity.hpp>
#include <xpool/devkit/timeline/transportruntime.hpp>
#include <xpool/devkit/transport_observer.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_devkit(py::module_ &module) {
  auto devkit = module.def_submodule("devkit", "Development-only native observation interfaces.");
  auto fabric = devkit.def_submodule("fabric_observer", "Per-PE Fabric protocol observation.");
  auto graph = devkit.def_submodule("graph_observer", "FfnAgent CUDA Graph observation.");
  auto routing = devkit.def_submodule("ffn_routing_observer", "FfnAgent routing observation.");
  auto transport = devkit.def_submodule("transport_observer", "Per-process Transport protocol observation.");

  bind_timeline(devkit);

  namespace timeline = xpool::devkit::timeline;
  timeline::install_routing_encoder(
      +[](timeline::BatchSource source, std::span<const timeline::RoutingAttachment> records) -> std::string {
        const auto gil = py::gil_scoped_acquire{};
        auto values = py::list{};
        for (const auto &record : records)
          values.append(py::cast(record));
        const auto uuid = timeline::format_identity_uuid(source.run_id);
        return py::module_::import("xpool.devkit.ffn_routing_observer")
            .attr("encode_routing")(std::string{uuid.data(), uuid.size()}, source.producer, values)
            .cast<std::string>();
      });
  auto payload = py::class_<timeline::RoutingAttachment>(routing, "RoutingAttachment",
                                                         "Immutable Writer-owned Routing attachment.");
  payload.def_readonly("sequence", &timeline::RoutingAttachment::sequence, "Original attachment sequence.");
  payload.def_readonly("generation", &timeline::RoutingAttachment::generation, "Original attachment generation.");
  payload.def_readonly("instance_index", &timeline::RoutingAttachment::instance_index,
                       "Original attachment instance_index.");
  payload.def_readonly("invocation_sequence", &timeline::RoutingAttachment::invocation_sequence,
                       "Original attachment invocation_sequence.");
  payload.def_readonly("lane_index", &timeline::RoutingAttachment::lane_index, "Original attachment lane_index.");
  payload.def_readonly("lease_sequence", &timeline::RoutingAttachment::lease_sequence,
                       "Original attachment lease_sequence.");
  payload.def_readonly("layer_ordinal", &timeline::RoutingAttachment::layer_ordinal,
                       "Original attachment layer_ordinal.");
  payload.def_readonly("pe", &timeline::RoutingAttachment::pe, "Original attachment pe.");
  payload.def_readonly("clock_domain_id", &timeline::RoutingAttachment::clock_domain_id,
                       "Original attachment clock_domain_id.");
  payload.def_readonly("raw_publication_time_ns", &timeline::RoutingAttachment::raw_publication_time_ns,
                       "Original attachment raw_publication_time_ns.");
  payload.def_readonly("topk_ids", &timeline::RoutingAttachment::topk_ids, "Original attachment topk_ids.");
  payload.def_readonly("topk_weights", &timeline::RoutingAttachment::topk_weights, "Original attachment topk_weights.");
  graph.def("record_host", &timeline::record_host_graph, py::arg("kind"), py::arg("phase"), py::arg("backend"),
            py::arg("identity"), py::arg("template_sequence"), py::arg("time_ns"),
            "Submit an actual Host Graph boundary without file I/O.");
  graph.def("finish_attachments", &timeline::finish_attachments, py::arg("producers_retired"),
            "Join attachments after actual runtime retirement.", py::call_guard<py::gil_scoped_release>());
  graph.def("prepare_attachments", &timeline::prepare_attachments, py::arg("host"), py::arg("device"),
            py::arg("ordinal"), py::arg("generation"), py::arg("graph_outdir"), py::arg("routing_outdir"),
            py::arg("routing_demand"), "Prepare registered attachment Sources before collection.",
            py::call_guard<py::gil_scoped_release>());
  routing.def("collect_drained", &xpool::devkit::ffn_routing_observer::collect_drained,
              "Queue Routing payload after proven producer drain.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "prepare",
      [](std::shared_ptr<xpool::devkit::timeline::MetadataRegistry> metadata, std::uint64_t generation,
         const xpool::fabric::ArenaProjection &projection, int pe, const std::string &outdir) {
        auto &runtime = xpool::RuntimeState::singleton();
        runtime.require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                             "xpool.native.devkit.fabric_observer.prepare");
        TORCH_CHECK(pe >= 0 && ((runtime.role() == xpool::RuntimeRole::AtnAgent) ==
                                (static_cast<std::size_t>(pe) < projection.atnagent_count)),
                    "xpool Fabric observation PE differs from the initialized runtime role");
        xpool::devkit::fabric_observer::prepare(std::move(metadata), generation, projection, pe,
                                                runtime.device("xpool.native.devkit.fabric_observer.prepare"), outdir);
      },
      py::arg("metadata"), py::arg("generation"), py::arg("projection"), py::arg("pe"), py::arg("outdir"),
      "Admit the actual participant before join; install its view only at real join.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def("cancel_preparation", &xpool::devkit::fabric_observer::cancel_preparation,
             "Release never-activated cold preparation; false retains an owner needing production retirement.",
             py::call_guard<py::gil_scoped_release>());
  fabric.def("finish_retirement", &xpool::devkit::fabric_observer::finish_retirement,
             "Join observation workers after Device retirement without renewing the deadline; false retains storage.",
             py::call_guard<py::gil_scoped_release>());
  fabric.def("allocation_bytes", &xpool::devkit::fabric_observer::allocation_bytes, py::arg("instance_count"),
             py::arg("executor_lane_count"), "Return the optional process-local Fabric Observer allocation size.");

  py::class_<xpool::devkit::ffn_routing_observer::MemoryDemand>(routing, "MemoryDemand",
                                                                "Complete Routing Device and Host coexistence demand.")
      .def(py::init<>())
      .def_readonly("device_bytes", &xpool::devkit::ffn_routing_observer::MemoryDemand::device_bytes,
                    "Actual Device allocation requirement.")
      .def_readonly("host_bytes", &xpool::devkit::ffn_routing_observer::MemoryDemand::host_bytes,
                    "Complete retained copy and serialization coexistence requirement.");
  routing.def("memory_demand", &xpool::devkit::ffn_routing_observer::memory_demand, py::arg("projection"),
              py::arg("ffnagent_index"),
              "Compute complete allocation demand from the actual placed projection; no CUDA.");
  routing.def("allocation_bytes", &xpool::devkit::ffn_routing_observer::allocation_bytes, py::arg("element_capacity"),
              "Return the optional process-local Routing Observer allocation size.");

  transport.def("prepare_endpoint", &xpool::devkit::transport_observer::prepare_endpoint, py::arg("metadata"),
                py::arg("region"), py::arg("instance_index"), py::arg("rank"), py::arg("device"), py::arg("atnagent"),
                py::arg("path"), "Admit and install one registered Transport Endpoint before publication/attachment.",
                py::call_guard<py::gil_scoped_release>());
  transport.def("finish_retirement", &xpool::devkit::transport_observer::finish_retirement,
                "Join after actual endpoint close; false retains unfinished owners.",
                py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
