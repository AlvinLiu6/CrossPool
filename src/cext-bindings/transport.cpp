#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/stl.h>
#include <torch/python.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/preparation.hpp>
#include <xpool/ffn.hpp>
#include <xpool/runtime.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/instance.hpp>

namespace py = pybind11;

namespace xpool::bindings {
namespace {

void require_transport_pool(const std::shared_ptr<xpool::transport::ObservationPool> &pool) {
  auto &preparation = xpool::devkit::timeline::ProcessPreparation::singleton();
  const auto policy = preparation.read();
  const auto required = policy && policy->transport_identity_required();
  TORCH_CHECK(static_cast<bool>(pool) == required,
              "xpool Transport observation pool disagrees with immutable bootstrap policy");
  if (pool) {
    auto &runtime = xpool::RuntimeState::singleton();
    TORCH_CHECK(preparation.prepare_transport(pool->source(), runtime.device("xpool transport observation"),
                                              runtime.role() == xpool::RuntimeRole::AtnAgent) == pool,
                "xpool Transport requires the unique prepared observation pool");
  }
}

} // namespace

void bind_transport(py::module_ &module) {
  auto transport = module.def_submodule("transport", "Native CUDA IPC transport control and lifecycle functions.");
  transport.attr("ARENA_HANDLE_HEX_LENGTH") = xpool::transport::ArenaHandle::encoded_size;
  transport.attr("OBSERVATION_ALLOCATION_BYTES") = xpool::transport::ObservationPool::allocation_bytes;
  py::class_<xpool::transport::ObservationPool, std::shared_ptr<xpool::transport::ObservationPool>>(
      transport, "ObservationPool", "Process-lifetime bounded IPC owner; created only by common preparation.");
  py::class_<xpool::transport::ObservationRegion>(transport, "ObservationRegion", "Immutable C0 v4 IPC descriptor.")
      .def(py::init([](std::uint32_t version, std::uint64_t owner, std::uint64_t sequence, std::uint64_t generation,
                       std::uint64_t allocation_bytes, const std::string &handle) {
             auto value = xpool::transport::ObservationRegion{
                 version, owner, sequence, generation, allocation_bytes, xpool::transport::ArenaHandle::decode(handle)};
             xpool::transport::validate_observation_region(value, true, generation, owner);
             return value;
           }),
           py::arg("version"), py::arg("owner_producer"), py::arg("endpoint_sequence"), py::arg("generation"),
           py::arg("allocation_bytes"), py::arg("handle"))
      .def_readonly("version", &xpool::transport::ObservationRegion::version, "C0 descriptor version, exactly four.")
      .def_readonly("owner_producer", &xpool::transport::ObservationRegion::owner_producer,
                    "Actual AtnAgent Device Producer.")
      .def_readonly("endpoint_sequence", &xpool::transport::ObservationRegion::endpoint_sequence,
                    "Positive allocation identity, never reused.")
      .def_readonly("generation", &xpool::transport::ObservationRegion::generation,
                    "Registered production Generation reference.")
      .def_readonly("allocation_bytes", &xpool::transport::ObservationRegion::allocation_bytes,
                    "Owned extent including alignment, in bytes.")
      .def_property_readonly(
          "handle", [](const xpool::transport::ObservationRegion &value) { return value.handle.encode(); },
          "Canonical CUDA IPC handle; control plane only.");
  transport.def(
      "create_arena",
      [](std::size_t instance_index, std::size_t instance_rank, std::size_t payload_row_capacity,
         std::size_t hidden_size, const py::object &dtype, std::size_t atn_tp_rank, std::size_t atn_tp_size,
         std::size_t atn_dp_rank, std::size_t atn_dp_size,
         std::shared_ptr<xpool::transport::ObservationPool> observation_pool, std::uint64_t generation) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.create_arena");
        const auto payload_dtype = torch::python::detail::py_object_to_dtype(dtype);
        TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
                    "xpool Transport payload dtype must be torch.float16 or torch.bfloat16");
        const auto release = py::gil_scoped_release{};
        require_transport_pool(observation_pool);
        return xpool::transport::AtnAgentRuntime::singleton()
            .create_arena(xpool::RuntimeState::singleton().device("xpool.native.transport.create_arena"),
                          instance_index, instance_rank, payload_row_capacity, hidden_size, payload_dtype, atn_tp_rank,
                          atn_tp_size, atn_dp_rank, atn_dp_size, std::move(observation_pool), generation)
            .encode();
      },
      py::arg("instance_index"), py::arg("instance_rank"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
      py::arg("dtype"), py::arg("atn_tp_rank"), py::arg("atn_tp_size"), py::arg("atn_dp_rank"), py::arg("atn_dp_size"),
      py::arg("observation_pool") = py::none(), py::arg("generation") = 0,
      "Create one AtnAgent-owned Arena and its policy-required observation allocation.");
  transport.def(
      "observation_region",
      [](const std::string &handle) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.observation_region");
        return xpool::transport::AtnAgentRuntime::singleton().observation_region(
            xpool::transport::ArenaHandle::decode(handle));
      },
      py::arg("handle"), "Read the immutable descriptor of an owned Arena; no live Device read.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "activate",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent, "xpool.native.transport.activate");
        xpool::transport::AtnAgentRuntime::singleton().activate();
      },
      "Launch resident Transport kernels for every created arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "check_health",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.check_health");
        xpool::transport::AtnAgentRuntime::singleton().check_health();
      },
      "Raise when an active Transport resident has failed.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_async",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_async");
        xpool::transport::AtnAgentRuntime::singleton().drain_async();
      },
      "Begin asynchronous drain for every active Transport resident.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_pending",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_pending");
        return xpool::transport::AtnAgentRuntime::singleton().drain_pending();
      },
      "Return whether any Transport resident drain remains pending.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "destroy_arenas",
      [](const std::vector<std::string> &handles, bool peer_mappings_detached) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.destroy_arenas");
        auto decoded = std::vector<xpool::transport::ArenaHandle>{};
        decoded.reserve(handles.size());
        for (const auto &handle : handles) {
          decoded.push_back(xpool::transport::ArenaHandle::decode(handle));
        }
        xpool::transport::AtnAgentRuntime::singleton().destroy_arenas(decoded, peer_mappings_detached);
      },
      py::arg("handles"), py::arg("peer_mappings_detached") = false,
      "Destroy drained owned Arenas after actual peer detach, or before any publication attempt.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "attach_arena",
      [](std::size_t instance_index, std::size_t rank, const std::string &handle,
         std::shared_ptr<xpool::transport::ObservationPool> observation_pool,
         const std::optional<xpool::transport::ObservationRegion> &descriptor, std::uint64_t generation,
         std::uint64_t owner_producer) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.attach_arena");
        require_transport_pool(observation_pool);
        xpool::transport::InstanceRankRuntime::singleton().attach_arena(
            instance_index, rank, xpool::transport::ArenaHandle::decode(handle), std::move(observation_pool),
            descriptor, generation, owner_producer);
      },
      py::arg("instance_index"), py::arg("rank"), py::arg("handle"), py::arg("observation_pool") = py::none(),
      py::arg("descriptor") = py::none(), py::arg("generation") = 0, py::arg("owner_producer") = 0,
      "Attach one Instance-rank process to its rank-local CUDA IPC arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "detach_arena",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.detach_arena");
        xpool::transport::InstanceRankRuntime::singleton().detach_arena();
      },
      "Detach the Instance-rank process from its current CUDA IPC arena.", py::call_guard<py::gil_scoped_release>());
  transport.def(
      "read_generation_failure",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.read_generation_failure");
        return xpool::transport::InstanceRankRuntime::singleton().read_generation_failure();
      },
      "Return the attached arena's generation-wide failure code.", py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
