#pragma once

/// \file xpool/devkit/transport_observer.hpp
/// \brief Process-local Transport observation records and Host readout.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <xpool/abort.hpp>
#include <xpool/devkit/timeline/resource.hpp>
#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>
#include <xpool/transport/hooks.hpp>
#include <xpool/transport/protocol.hpp>
#include <xpool/utils/trace.hpp>

namespace xpool::devkit::timeline::detail {
struct LegacyRecordAccess;
}

namespace xpool::devkit::transport_observer {

/// Immutable request facts and dependency-checked timestamps for one request.
class alignas(8) Record {
public:
  /// Return the positive process-local sequence assigned at record creation.
  XPOOL_HOST_DEVICE_FN std::uint64_t trace_id() const { return trace_id_; }
  /// Return the physical hidden-state row count carried by the request.
  XPOOL_HOST_DEVICE_FN std::size_t payload_rows() const { return payload_rows_; }
  /// Return the model-local FFN layer ordinal.
  XPOOL_HOST_DEVICE_FN std::size_t layer_ordinal() const { return request_.layer_ordinal; }
  /// Return the forward mode carried by the request.
  XPOOL_HOST_DEVICE_FN xpool::ffn::ForwardMode forward_mode() const { return request_.forward_mode; }
  /// Return the output requirement carried by the request.
  XPOOL_HOST_DEVICE_FN xpool::ffn::OutputRequirement output_requirement() const { return request_.output_requirement; }
  /// Return the physical DP row layout carried by the request.
  XPOOL_HOST_DEVICE_FN xpool::ffn::DpRowLayout dp_row_layout() const { return request_.dp_row_layout; }
  /// Return the effective request result retained by this trace.
  XPOOL_HOST_DEVICE_FN xpool::ffn::ResultCode result_code() const { return result_code_; }
  /// Test whether one event timestamp has been recorded.
  XPOOL_HOST_DEVICE_FN bool recorded(xpool::hooks::TransportProtocolEventKind event) const {
    return timeline_.recorded(event);
  }
  /// Return one GPU global-timer timestamp in nanoseconds.
  /// Values are comparable only inside the same GPU clock domain.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(xpool::hooks::TransportProtocolEventKind event) const {
    return timeline_.timestamp(event);
  }

#if defined(__CUDACC__)
  /// Initialize a reserved record and mark staging start.
  XPOOL_DEVICE_FN void begin_instance(std::uint64_t trace_id, std::size_t payload_rows,
                                      const xpool::transport::RequestMetadata &request);

  /// Initialize one AtnAgent-local record after observing publication.
  XPOOL_DEVICE_FN void begin_atnagent(std::uint64_t trace_id, std::size_t payload_rows,
                                      const xpool::transport::RequestMetadata &request);

  /// Record completion of cooperative request staging.
  XPOOL_DEVICE_FN void request_staging_completed();

  /// Record completed request publication.
  XPOOL_DEVICE_FN void request_published();

  /// Record entry into AtnAgent request execution.
  XPOOL_DEVICE_FN void execution_started();

  /// Record completion of the selected execution path.
  XPOOL_DEVICE_FN void execution_completed();

  /// Record the completed AtnAgent result publication.
  XPOOL_DEVICE_FN void result_published(xpool::ffn::ResultCode result_code);

  /// Record the Instance rank's effective result after canonical-failure precedence.
  XPOOL_DEVICE_FN void result_observed(xpool::ffn::ResultCode effective_result_code);

  /// Record completion of successful output copying.
  XPOOL_DEVICE_FN void output_copied();

  /// Record completed successful acknowledgement and mailbox reuse.
  XPOOL_DEVICE_FN void result_acknowledged();

  /// Record terminal closure of an owned Staging request.
  XPOOL_DEVICE_FN void closed(xpool::ffn::ResultCode result_code);

  /// Record terminal closure after observing an Evaluated result.
  XPOOL_DEVICE_FN void closed();
#endif

private:
  friend struct xpool::devkit::timeline::detail::LegacyRecordAccess;
  std::uint64_t trace_id_ = 0;
  std::size_t payload_rows_ = 0;
  xpool::transport::RequestMetadata request_{};
  xpool::ffn::ResultCode result_code_ = xpool::ffn::ResultCode::ProtocolMismatch;
  xpool::utils::trace::Timeline<xpool::hooks::TransportProtocolEventKind> timeline_{};
};

/// Internal storage charge copied with each independent compatibility vector.
/// Declared before vectors so the charge outlives their actual allocation.
struct SnapshotStorage {
  /// Shared process ledger; no charge is returned merely because files exist.
  std::shared_ptr<timeline::MemoryBudget> budget;
  /// Exact admitted vector/control capacity.
  timeline::MemoryBudget::Reservation charge;
  SnapshotStorage() = default;
  /// Admit before allocating the accompanying vectors.
  SnapshotStorage(std::shared_ptr<timeline::MemoryBudget> owner, std::size_t bytes)
      : budget(std::move(owner)), charge(budget->reserve(timeline::MemoryKind::Snapshot, bytes)) {}
  /// Independent copies must admit their own storage before vector copying.
  SnapshotStorage(const SnapshotStorage &other) : budget(other.budget) {
    if (budget)
      charge = budget->reserve(timeline::MemoryKind::Snapshot, other.charge.bytes());
  }
  /// Transfer the retained charge without a second allocation or admission.
  SnapshotStorage(SnapshotStorage &&) noexcept = default;
  /// Assignment is used only alongside a complete vector swap/move.
  SnapshotStorage &operator=(SnapshotStorage &&other) noexcept {
    if (this != &other) {
      charge = std::move(other.charge);
      budget = std::move(other.budget);
    }
    return *this;
  }
};

/// One endpoint's process-local bounded Transport observation.
struct EndpointSnapshot {
  EndpointSnapshot() = default;
  /// Admit independent storage before copying the Record vector.
  EndpointSnapshot(const EndpointSnapshot &) = default;
  /// Transfer the Record vector together with its existing storage charge.
  EndpointSnapshot(EndpointSnapshot &&) noexcept = default;
  /// Copy admission precedes mutation; old vectors die before their charges.
  EndpointSnapshot &operator=(EndpointSnapshot copy) noexcept {
    using std::swap;
    swap(storage, copy.storage);
    swap(instance_index, copy.instance_index);
    swap(instance_rank, copy.instance_rank);
    swap(sequence, copy.sequence);
    swap(dropped, copy.dropped);
    swap(records, copy.records);
    return *this;
  }
  /// Retained charge for this independent Record vector.
  SnapshotStorage storage;
  /// Fabric instance addressed by the endpoint.
  std::size_t instance_index = 0;
  /// Rank within the instance's Transport group.
  std::size_t instance_rank = 0;
  /// Total number of trace reservations attempted.
  std::uint64_t sequence = 0;
  /// Reservations omitted after capacity was exhausted.
  std::uint64_t dropped = 0;
  /// Retained records in sequence order.
  std::vector<Record> records;
};

/// Process-local observation for every open Transport endpoint.
struct Snapshot {
  Snapshot() = default;
  /// Admit independent endpoint and Record vectors before copying them.
  Snapshot(const Snapshot &) = default;
  /// Transfer all vectors and their retained charges together.
  Snapshot(Snapshot &&) noexcept = default;
  /// Keep vector lifetime and budget ownership together on every assignment.
  Snapshot &operator=(Snapshot copy) noexcept {
    using std::swap;
    swap(storage, copy.storage);
    swap(endpoints, copy.endpoints);
    return *this;
  }
  /// Retained charge for the Endpoint vector.
  SnapshotStorage storage;
  /// Snapshots for every endpoint currently open in this process.
  std::vector<EndpointSnapshot> endpoints;
};

/// Return the current process-local Transport snapshot when observation is enabled.
/// \pre Every related endpoint is quiesced or otherwise synchronized.
/// \return Host-owned endpoint collection, or no value when no endpoint is open.
/// \throws c10::Error when CUDA copy or retained observer state is invalid.
std::optional<Snapshot> read();

} // namespace xpool::devkit::transport_observer
