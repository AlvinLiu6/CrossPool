#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;
namespace timeline = xpool::devkit::timeline;

namespace xpool::bindings {

void bind_timeline_policy(py::module_ &module) {
  auto limits = py::class_<timeline::PolicyLimits>(module, "PolicyLimits", "Immutable numeric collection policy.");
  limits
      .def_readonly("device_bytes", &timeline::PolicyLimits::device_bytes,
                    "Complete process/Device allocation ceiling, in bytes.")
      .def_readonly("host_bytes", &timeline::PolicyLimits::host_bytes,
                    "Internal process Host allocation ceiling, in bytes.")
      .def_readonly("chunk_bytes", &timeline::PolicyLimits::chunk_bytes,
                    "Maximum complete chunk size including its header, in bytes.")
      .def_readonly("collection_interval_ms", &timeline::PolicyLimits::collection_interval_ms,
                    "Background collection cadence, in milliseconds.")
      .def_readonly("disk_bytes", &timeline::PolicyLimits::disk_bytes, "Shared process artifact quota, in bytes.")
      .def_readonly("metadata_reserve_bytes", &timeline::PolicyLimits::metadata_reserve_bytes,
                    "Protected metadata and close-report space within the disk quota, in bytes.")
      .def_readonly("shutdown_wait_ms", &timeline::PolicyLimits::shutdown_wait_ms,
                    "Maximum additional observation wait within the production deadline, in milliseconds.")
      .def_readonly("calibration_block_attempts", &timeline::PolicyLimits::calibration_block_attempts,
                    "Fixed C0 calibration attempts per block.")
      .def_readonly("device_event_slot_capacity", &timeline::PolicyLimits::device_event_slot_capacity,
                    "Immutable Device Event slot count.")
      .def_readonly("endpoint_capacity", &timeline::PolicyLimits::endpoint_capacity,
                    "Maximum simultaneously open Operation IPC identity regions.")
      .def_readonly("context_slot_capacity", &timeline::PolicyLimits::context_slot_capacity,
                    "Dynamic Context slots owned by the Graph runtime.")
      .def_readonly("sampler_slot_capacity", &timeline::PolicyLimits::sampler_slot_capacity,
                    "Raw Device clock sampler slot count.");
  py::class_<timeline::CollectionPolicy>(module, "CollectionPolicy", "Copy of the installed process policy.")
      .def_readonly("limits", &timeline::CollectionPolicy::limits,
                    "Immutable numeric projection of validated xpool.config options.")
      .def_readonly("unified", &timeline::CollectionPolicy::unified, "Whether the common raw timeline was requested.")
      .def_property_readonly("enabled", &timeline::CollectionPolicy::enabled,
                             "Whether any independent or unified observation class is requested.")
      .def_property_readonly("transport_events", &timeline::CollectionPolicy::transport_events,
                             "Whether policy requests raw Transport events.")
      .def_property_readonly("fabric_events", &timeline::CollectionPolicy::fabric_events,
                             "Whether policy requests raw Fabric events.")
      .def_property_readonly("graph_events", &timeline::CollectionPolicy::graph_events,
                             "Whether policy requests raw Graph events.")
      .def_property_readonly("routing_events", &timeline::CollectionPolicy::routing_events,
                             "Whether policy requests routing evidence.")
      .def_property_readonly("transport_identity_required", &timeline::CollectionPolicy::transport_identity_required,
                             "Whether a configured consumer requires shared Transport identity.");
  module.def(
      "configure_policy",
      [](const timeline::Options &options, const xpool::debug::Options &legacy, py::bytes configuration_fingerprint,
         bool explicit_transport_identity_consumer) {
        const auto bytes = static_cast<std::string>(configuration_fingerprint);
        if (bytes.size() != 32)
          throw py::value_error("configuration fingerprint requires 32 SHA256 bytes");
        auto digest = std::array<std::byte, 32>{};
        std::transform(bytes.begin(), bytes.end(), digest.begin(),
                       [](char value) { return static_cast<std::byte>(value); });
        timeline::ProcessPreparation::singleton().configure(
            timeline::make_collection_policy(options, legacy, digest, explicit_transport_identity_consumer));
      },
      py::arg("options"), py::arg("legacy"), py::arg("configuration_fingerprint"),
      py::arg("explicit_transport_identity_consumer") = false,
      "Install immutable xpool.config policy before Native initialization; no CUDA, threads or I/O.");
  module.def(
      "read_policy", []() { return timeline::ProcessPreparation::singleton().read(); },
      "Copy installed policy or return None.");
  module.def(
      "limit_retirement",
      [](std::uint64_t deadline_ns) { timeline::ProcessPreparation::singleton().limit_retirement(deadline_ns); },
      py::arg("deadline_ns"),
      "Tighten the process observation deadline using production CLOCK_MONOTONIC nanoseconds; no CUDA.");
  py::class_<timeline::ProcessBudgets>(module, "ProcessBudgets", "Common process Host and disk ledgers.")
      .def_readonly("host", &timeline::ProcessBudgets::host,
                    "Common Host ledger retained by this view and shared with metadata and Graph.")
      .def_readonly("device", &timeline::ProcessBudgets::device,
                    "Common Device ledger, or None until a Device provider is prepared.");
  module.def(
      "prepare_transport",
      [](timeline::BatchSource source, int logical_device) {
        auto &runtime = xpool::RuntimeState::singleton();
        runtime.require_role({xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent},
                             "xpool.native.devkit.timeline.prepare_transport");
        TORCH_CHECK(runtime.device("xpool.native.devkit.timeline.prepare_transport") == logical_device,
                    "xpool observation registration differs from the initialized logical Device");
        return timeline::ProcessPreparation::singleton().prepare_transport(
            source, runtime.device("xpool.native.devkit.timeline.prepare_transport"),
            runtime.role() == xpool::RuntimeRole::AtnAgent);
      },
      py::arg("source"), py::arg("logical_device"),
      "Admit the unique local Device observation pool after checking its actual ordinal; no CUDA allocation.");
  module.def(
      "prepare_budgets",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
            "xpool.native.devkit.timeline.prepare_budgets");
        return timeline::ProcessPreparation::singleton().budgets();
      },
      "Borrow the observed process ledgers after bootstrap; no CUDA, workers or files.");
}

} // namespace xpool::bindings
