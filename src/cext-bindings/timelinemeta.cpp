#include <stdexcept>

#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/metaregistry.hpp>

namespace py = pybind11;
namespace timeline = xpool::devkit::timeline;

namespace xpool::bindings {
void bind_timeline_meta(py::module_ &module) {
  py::enum_<timeline::MetadataCapability>(module, "MetadataCapability",
                                          "C0 provider availability, independent of configuration enablement.")
      .value("enabled", timeline::MetadataCapability::Enabled)
      .value("disabled", timeline::MetadataCapability::Disabled)
      .value("not_applicable", timeline::MetadataCapability::NotApplicable)
      .value("unavailable", timeline::MetadataCapability::Unavailable);
  py::class_<timeline::MetadataCapabilities>(module, "MetadataCapabilities",
                                             "Actual immutable capture/path declarations.")
      .def(py::init<>())
      .def_readwrite("transport", &timeline::MetadataCapabilities::transport,
                     "Actual transport provider capability, independent of the requested policy.")
      .def_readwrite("fabric", &timeline::MetadataCapabilities::fabric,
                     "Actual fabric provider capability, independent of the requested policy.")
      .def_readwrite("graph", &timeline::MetadataCapabilities::graph,
                     "Actual graph provider capability, independent of the requested policy.")
      .def_readwrite("routing", &timeline::MetadataCapabilities::routing,
                     "Actual routing provider capability, independent of the requested policy.")
      .def_readwrite("calibration", &timeline::MetadataCapabilities::calibration,
                     "Actual calibration provider capability, independent of the requested policy.");
  py::class_<timeline::MetadataLimits>(module, "MetadataLimits", "Fixed semantic metadata table/document bounds.")
      .def(py::init<>())
      .def_readwrite("producers", &timeline::MetadataLimits::producers,
                     "Maximum Host plus Device Producer definitions in one complete document.")
      .def_readwrite("clocks", &timeline::MetadataLimits::clocks, "Maximum referenced raw clock domains.")
      .def_readwrite("generations", &timeline::MetadataLimits::generations,
                     "Maximum retained immutable production Generation definitions.")
      .def_readwrite("models_per_generation", &timeline::MetadataLimits::models_per_generation,
                     "Maximum model and topology entries per Generation.")
      .def_readwrite("endpoints", &timeline::MetadataLimits::endpoints,
                     "Maximum referenced Transport Endpoint lifetimes.")
      .def_readwrite("templates", &timeline::MetadataLimits::templates,
                     "Maximum Graph template definitions across retained lifetimes.")
      .def_readwrite("compute_templates", &timeline::MetadataLimits::compute_templates,
                     "Maximum compute-template references in one lane template.")
      .def_readwrite("artifacts", &timeline::MetadataLimits::artifacts,
                     "Maximum immutable artifact definitions for this Producer.")
      .def_readwrite("document_bytes", &timeline::MetadataLimits::document_bytes,
                     "Complete UTF-8 document capacity including its final newline.");
  py::class_<timeline::MetadataRegistry, std::shared_ptr<timeline::MetadataRegistry>>(
      module, "MetadataRegistry", "Source-bound typed metadata, fixed serialization and exact-revision coverage.")
      .def(py::init(&timeline::MetadataRegistry::create), py::arg("host"), py::arg("source"), py::arg("policy"),
           py::arg_v("capabilities", timeline::MetadataCapabilities{}, "MetadataCapabilities()"),
           py::arg_v("limits", timeline::MetadataLimits{}, "MetadataLimits()"))
      .def_property_readonly("source", &timeline::MetadataRegistry::source, py::return_value_policy::reference_internal,
                             "Borrowed registered provenance retained by this metadata owner.")
      .def("add_host", &timeline::MetadataRegistry::add_host, py::arg("process"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_device", &timeline::MetadataRegistry::add_device, py::arg("process"), py::arg("source"),
           py::arg("logical_device"), py::arg("physical_device_uuid"), py::arg("raw_epoch"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_generation", &timeline::MetadataRegistry::add_generation, py::arg("id"), py::arg("fabric_generation"),
           py::arg("atnagent_count"), py::arg("ffnagent_count"), py::arg("fifo"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_pe_member", &timeline::MetadataRegistry::add_pe_member, py::arg("generation"), py::arg("pe"),
           py::arg("host_producer"), py::arg("device_producer"), py::call_guard<py::gil_scoped_release>())
      .def("fabric_pe", &timeline::MetadataRegistry::fabric_pe, py::arg("generation"), py::arg("producer"),
           py::call_guard<py::gil_scoped_release>())
      .def("add_model", &timeline::MetadataRegistry::add_model, py::arg("generation"), py::arg("instance_index"),
           py::arg("model_id"), py::arg("atn_tp_size"), py::arg("atn_dp_size"), py::arg("ffn_tp_size") = py::none(),
           py::call_guard<py::gil_scoped_release>())
      .def("add_endpoint", &timeline::MetadataRegistry::add_endpoint, py::arg("owner"), py::arg("sequence"),
           py::arg("generation"), py::arg("instance_index"), py::arg("instance_rank"), py::arg("model_id"),
           py::arg("arena_suffix"), py::arg("instance_producer"), py::arg("supersedes") = py::none(),
           py::call_guard<py::gil_scoped_release>())
      .def("validate_preparation", &timeline::MetadataRegistry::validate_preparation, py::arg("device_index"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "ensure_revision", [](timeline::MetadataRegistry &metadata) { return metadata.ensure_revision().revision; },
          "Publish newly registered facts to the bound catalog before their first event; equal retries reuse its "
          "revision.",
          py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
