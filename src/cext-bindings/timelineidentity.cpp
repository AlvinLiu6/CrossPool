#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/identity.hpp>

namespace py = pybind11;
namespace timeline = xpool::devkit::timeline;

namespace xpool::bindings {
namespace {
py::str uuid_text(const timeline::IdentityUuid &uuid) {
  const auto text = timeline::format_identity_uuid(uuid);
  return py::str(text.data(), text.size());
}
} // namespace

void bind_timeline_identity(py::module_ &module) {
  py::class_<timeline::ProcessIdentity>(module, "ProcessIdentity", "Actual immutable launch facts.")
      .def(py::init([](const std::string &creation, const std::string &host_creation, std::uint64_t pid,
                       xpool::RuntimeRole role) {
             return timeline::ProcessIdentity{timeline::parse_identity_uuid(creation),
                                              timeline::parse_identity_uuid(host_creation), pid, role};
           }),
           py::arg("creation_id"), py::arg("host_creation_id"), py::arg("pid"), py::arg("runtime_role"))
      .def_property_readonly(
          "creation_id", [](const timeline::ProcessIdentity &value) { return uuid_text(value.creation_id); },
          "UUID assigned to one actual process creation.")
      .def_property_readonly(
          "host_creation_id", [](const timeline::ProcessIdentity &value) { return uuid_text(value.host_creation_id); },
          "Actual Host boot UUID defining the monotonic clock epoch.")
      .def_readonly("pid", &timeline::ProcessIdentity::pid,
                    "Actual OS process id; process creation identity remains separate.")
      .def_readonly("runtime_role", &timeline::ProcessIdentity::runtime_role,
                    "Actual observed runtime role; Daemon has no event Producer.");
  py::class_<timeline::RegisteredProcess>(module, "RegisteredProcess",
                                          "One process creation and its unique Host Source.")
      .def(py::init([](timeline::ProcessIdentity process, timeline::BatchSource host) {
             return timeline::RegisteredProcess{process, host};
           }),
           py::arg("process"), py::arg("host"))
      .def_readonly("process", &timeline::RegisteredProcess::process,
                    "Immutable assigned process-creation and Host-boot facts.")
      .def_readonly("host", &timeline::RegisteredProcess::host, "The process Host Source assigned by the launcher.")
      .def_readonly("start_key", &timeline::RegisteredProcess::start_key,
                    "OS creation retry key; imported metadata-only values use zero.");
  py::class_<timeline::IdentityDirectory, std::shared_ptr<timeline::IdentityDirectory>>(
      module, "IdentityDirectory", "Bounded launcher-owned Run identity directory, with no CUDA or file I/O.")
      .def(py::init([](std::shared_ptr<timeline::MemoryBudget> host, const std::string &run, std::size_t processes,
                       std::size_t devices, std::size_t generations) {
             return timeline::IdentityDirectory::create(std::move(host), timeline::parse_identity_uuid(run), processes,
                                                        devices, generations);
           }),
           py::arg("host"), py::arg("run_id"), py::arg("process_capacity") = 64, py::arg("device_capacity") = 64,
           py::arg("generation_capacity") = 64, py::call_guard<py::gil_scoped_release>())
      .def(
          "assign_process",
          [](timeline::IdentityDirectory &owner, const std::string &host_creation, std::uint64_t pid,
             std::uint64_t start_key, xpool::RuntimeRole role) {
            return owner.assign_process(timeline::parse_identity_uuid(host_creation), pid, start_key, role);
          },
          py::arg("host_creation_id"), py::arg("pid"), py::arg("start_key"), py::arg("runtime_role"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "register_device",
          [](timeline::IdentityDirectory &owner, const std::string &creation, std::uint32_t logical_device,
             const std::string &physical_uuid, std::uint64_t epoch) {
            return owner.register_device(timeline::parse_identity_uuid(creation), logical_device, physical_uuid, epoch);
          },
          py::arg("process_creation_id"), py::arg("logical_device"), py::arg("physical_device_uuid"),
          py::arg("raw_epoch"), py::call_guard<py::gil_scoped_release>())
      .def("register_generation", &timeline::IdentityDirectory::register_generation, py::arg("fabric_generation"),
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &timeline::IdentityDirectory::stop, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("identity_exhausted", &timeline::IdentityDirectory::identity_exhausted,
                             "True after identity or fixed directory capacity exhaustion.");
}

} // namespace xpool::bindings
