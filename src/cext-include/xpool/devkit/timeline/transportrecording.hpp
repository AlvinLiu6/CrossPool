#pragma once

/// \file xpool/devkit/timeline/transportrecording.hpp
/// \brief Fixed per-Endpoint current facts for the shared Device event Source.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/devkit/timeline/view.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/transport/observation.hpp>
#include <xpool/transport/protocol.hpp>

namespace xpool::devkit::timeline {

/// One serialized mailbox owner's current request, never a history buffer.
struct TransportCurrent {
  xpool::transport::OperationFact operation{}; ///< Copied successful Claim identity.
  xpool::transport::RequestMetadata request{}; ///< Original values, including invalid enums.
  std::uint64_t payload_rows = 0;              ///< Original physical row count.
  xpool::ffn::ResultCode result = xpool::ffn::ResultCode::ProtocolMismatch; ///< Effective local result.
  bool result_known = false;                                                ///< Set only at an actual result boundary.

  /// Retain the actual result of this boundary, including a later shutdown
  /// after successful output copy. Other boundaries repeat the known result.
  XPOOL_HOST_DEVICE_FN void observe_result(xpool::hooks::TransportProtocolEventKind boundary,
                                           xpool::ffn::ResultCode observed) noexcept {
    using Kind = xpool::hooks::TransportProtocolEventKind;
    if (boundary == Kind::ResultPublished || boundary == Kind::ResultObserved || boundary == Kind::Closed) {
      result = observed;
      result_known = true;
    }
  }
};

/// Initialized/replaced only at the owning Endpoint's proven quiescence.
/// Device fields are accessed by its single protocol thread, never a live Host
/// reader. Other Endpoints may concurrently use their own table entries.
struct TransportRecordingEndpoint {
  std::uint64_t endpoint_owner = 0;    ///< Zero disables this slot.
  std::uint64_t endpoint_sequence = 0; ///< Registered lifetime, not table position.
  std::uint64_t generation = 0;        ///< Registered Generation reference.
  std::uint64_t instance_index = 0;    ///< Actual Transport arena index.
  std::uint32_t instance_rank = 0;     ///< Actual Transport rank.
  std::uint32_t role = 0;              ///< C0 Instance=1 or AtnAgent=2.
  std::uint64_t diagnostics = 0;       ///< Sticky local failures; no unbounded diagnostic log.
  TransportCurrent current{};          ///< Facts retained even when an earlier Event is lost.
};

/// Borrowed storage owned by DeviceChannel until actual Source retirement.
struct TransportRecordingView {
  BufferView buffer{};                             ///< The same Source used by this process's Fabric hooks.
  TransportRecordingEndpoint *endpoints = nullptr; ///< Fixed table, no Device allocation.
  std::size_t capacity = 0;                        ///< Admitted Endpoint slot count.
};

/// Complete rounded Device allocation; zero slots require no allocation.
/// Throws resource_budget on overflow before CUDA work.
std::size_t transport_recording_bytes(std::size_t capacity, std::uint64_t producer = 0);

static_assert(std::is_trivially_copyable_v<TransportRecordingEndpoint>);
static_assert(std::is_trivially_copyable_v<TransportRecordingView>);

} // namespace xpool::devkit::timeline
