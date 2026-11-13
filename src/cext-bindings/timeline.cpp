#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/identity.hpp>
#include <xpool/devkit/timeline/options.hpp>
#include <xpool/devkit/timeline/resource.hpp>

namespace py = pybind11;
namespace timeline = xpool::devkit::timeline;

namespace {
py::str uuid_text(const timeline::IdentityUuid &uuid) {
  const auto text = timeline::format_identity_uuid(uuid);
  return py::str(text.data(), text.size());
}
void bind_errors(py::module_ &module) {
  py::enum_<timeline::CollectionErrorCode>(module, "CollectionErrorCode", "Fixed C0 recovery categories.")
      .value("resource_budget", timeline::CollectionErrorCode::ResourceBudget)
      .value("allocation", timeline::CollectionErrorCode::Allocation)
      .value("transfer", timeline::CollectionErrorCode::Transfer)
      .value("invalid_release", timeline::CollectionErrorCode::InvalidRelease)
      .value("io", timeline::CollectionErrorCode::Io)
      .value("metadata_unavailable", timeline::CollectionErrorCode::MetadataUnavailable);
  // Use the same GIL-safe exception storage as pybind11's own registration,
  // preserving typed context instead of reducing CollectionError to what().
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::exception<timeline::CollectionError>> storage;
  storage.call_once_and_store_result(
      [&] { return py::exception<timeline::CollectionError>(module, "CollectionError", PyExc_RuntimeError); });
  py::register_local_exception_translator([](std::exception_ptr pointer) {
    if (!pointer)
      return;
    try {
      std::rethrow_exception(pointer);
    } catch (const timeline::CollectionError &error) {
      const auto type = py::reinterpret_borrow<py::object>(storage.get_stored());
      auto value = type(py::str(error.what()));
      value.attr("code") = py::cast(error.code());
      value.attr("producer") = py::int_(error.producer());
      value.attr("path") = error.path() ? py::cast(*error.path()) : py::none();
      value.attr("root_cause") = py::str(error.root_cause());
      PyErr_SetObject(type.ptr(), value.ptr());
    }
  });
}

} // namespace

namespace xpool::bindings {
void bind_timeline(py::module_ &devkit) {
  auto module = devkit.def_submodule("timeline", "Bounded observation cold preparation.");
  bind_errors(module);
  py::class_<timeline::MemoryReport>(module, "MemoryReport", "Shared internal collection allocation accounting.")
      .def_readonly("limit", &timeline::MemoryReport::limit, "Process internal collection admission ceiling, in bytes.")
      .def_readonly("used", &timeline::MemoryReport::used,
                    "Retained allocations and outstanding admission capacity, in bytes.")
      .def_readonly("peak", &timeline::MemoryReport::peak,
                    "Largest simultaneously charged byte total since ledger creation.")
      .def_readonly("categories", &timeline::MemoryReport::categories,
                    "Mutually exclusive category byte totals in Native MemoryKind order.");
  py::class_<timeline::MemoryBudget, std::shared_ptr<timeline::MemoryBudget>>(
      module, "MemoryBudget", "One admitted ledger, including its actual shared control allocation.")
      .def(py::init(&timeline::MemoryBudget::create), py::arg("limit"))
      .def("report", &timeline::MemoryBudget::report);
  auto options =
      py::class_<timeline::Options>(module, "Options", "Native policy projected from xpool.config before service.");
  options.def(py::init<>());
  options.def_readwrite("enable", &timeline::Options::enable, "Request all four unified evidence classes.");
  options.def_readwrite("outdir", &timeline::Options::outdir,
                        "Resolved unified artifact directory, paired with enable.");
  options.def_readwrite("device_bytes", &timeline::Options::device_bytes,
                        "Complete process/Device allocation ceiling, in bytes.");
  options.def_readwrite("host_bytes", &timeline::Options::host_bytes,
                        "Internal process Host allocation ceiling, in bytes.");
  options.def_readwrite("chunk_bytes", &timeline::Options::chunk_bytes,
                        "Maximum complete chunk size including its header, in bytes.");
  options.def_readwrite("collection_interval_ms", &timeline::Options::collection_interval_ms,
                        "Background collection cadence, in milliseconds.");
  options.def_readwrite("disk_bytes", &timeline::Options::disk_bytes, "Shared process artifact quota, in bytes.");
  options.def_readwrite("metadata_reserve_bytes", &timeline::Options::metadata_reserve_bytes,
                        "Protected metadata and close-report space within the disk quota, in bytes.");
  options.def_readwrite("shutdown_wait_ms", &timeline::Options::shutdown_wait_ms,
                        "Maximum additional observation wait within the production deadline, in milliseconds.");
  options.def_readwrite("calibration_block_attempts", &timeline::Options::calibration_block_attempts,
                        "Fixed C0 calibration attempts per block.");
  options.def_readwrite("device_event_slot_capacity", &timeline::Options::device_event_slot_capacity,
                        "Immutable Device Event slot count.");
  options.def_readwrite("endpoint_capacity", &timeline::Options::endpoint_capacity,
                        "Maximum simultaneously open Operation IPC identity regions.");
  options.def_readwrite("context_slot_capacity", &timeline::Options::context_slot_capacity,
                        "Dynamic Context slots owned by the Graph runtime.");
  options.def_readwrite("sampler_slot_capacity", &timeline::Options::sampler_slot_capacity,
                        "Raw Device clock sampler slot count.");
  py::class_<timeline::BatchSource>(module, "BatchSource", "Registered provenance; never inferred from Event fields.")
      .def(py::init([](std::string run_id, std::uint64_t producer, std::uint64_t clock_domain_id) {
             return timeline::BatchSource{timeline::parse_identity_uuid(run_id), producer, clock_domain_id};
           }),
           py::arg("run_id"), py::arg("producer"), py::arg("clock_domain_id"))
      .def_property_readonly(
          "run_id", [](const timeline::BatchSource &value) { return uuid_text(value.run_id); },
          "Canonical launcher-assigned Run UUID.")
      .def_readonly("producer", &timeline::BatchSource::producer,
                    "Positive launcher-assigned Producer identity within this Run.")
      .def_readonly("clock_domain_id", &timeline::BatchSource::clock_domain_id,
                    "Positive raw clock identity assigned to this Source.");
  bind_timeline_policy(module);
  bind_timeline_identity(module);
  bind_timeline_meta(module);
}
} // namespace xpool::bindings
