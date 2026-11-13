#include <algorithm>
#include <array>
#include <stdexcept>

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/timeline/legacyaccess.hpp>
#include <xpool/devkit/transport_observer.hpp>

namespace xpool::devkit::timeline::detail {

namespace {

bool available(const LegacyOperation &operation) {
  if ((operation.flags & static_cast<std::uint64_t>(LegacyFlag::Conflict)) != 0) {
    throw std::invalid_argument("xpool conflicting evidence cannot materialize a legacy Record");
  }
  return operation.local_trace_id != 0 &&
         (operation.flags & (static_cast<std::uint64_t>(LegacyFlag::MissingFacts) |
                             static_cast<std::uint64_t>(LegacyFlag::InvalidRequest))) == 0;
}

template <typename Event> auto timeline(const LegacyOperation &operation) {
  constexpr auto count = static_cast<std::size_t>(Event::Count);
  auto times = std::array<std::uint64_t, count>{};
  std::copy_n(operation.times.begin(), count, times.begin());
  return xpool::utils::trace::Timeline<Event>::from_retained(times, operation.recorded);
}

} // namespace

template <>
std::optional<xpool::devkit::transport_observer::Record>
LegacyRecordAccess::project<xpool::devkit::transport_observer::Record>(const LegacyOperation &operation,
                                                                       const LegacyScope &scope) {
  if (scope.family == LegacyFamily::Fabric)
    throw std::invalid_argument("xpool Transport Record requires an Endpoint view");
  if (!available(operation))
    return std::nullopt;
  auto result = xpool::devkit::transport_observer::Record{};
  result.trace_id_ = operation.local_trace_id;
  result.payload_rows_ = operation.facts[5];
  result.request_ = {.layer_ordinal = static_cast<std::size_t>(operation.facts[4]),
                     .forward_mode = static_cast<xpool::ffn::ForwardMode>(operation.facts[6]),
                     .output_requirement = static_cast<xpool::ffn::OutputRequirement>(operation.facts[7]),
                     .dp_row_layout = static_cast<xpool::ffn::DpRowLayout>(operation.facts[8])};
  if (!result.request_.valid())
    throw std::invalid_argument("xpool legacy request enums are invalid");
  result.result_code_ = (operation.known & (std::uint64_t{1} << 10)) != 0
                            ? static_cast<xpool::ffn::ResultCode>(operation.facts[10])
                            : xpool::ffn::ResultCode::ProtocolMismatch;
  result.timeline_ = timeline<xpool::hooks::TransportProtocolEventKind>(operation);
  return result;
}

template <>
std::optional<xpool::devkit::fabric_observer::Record>
LegacyRecordAccess::project<xpool::devkit::fabric_observer::Record>(const LegacyOperation &operation,
                                                                    const LegacyScope &scope) {
  if (scope.family != LegacyFamily::Fabric)
    throw std::invalid_argument("xpool Fabric Record requires a Generation view");
  if (!available(operation))
    return std::nullopt;
  auto result = xpool::devkit::fabric_observer::Record{};
  result.local_trace_id_ = operation.local_trace_id;
  result.key_ = {.instance_index = static_cast<std::size_t>(operation.facts[1]),
                 .invocation_sequence = operation.facts[2]};
  result.layer_ordinal_ = operation.facts[3];
  result.payload_rows_ = operation.facts[4];
  result.output_requirement_ = static_cast<xpool::ffn::OutputRequirement>(operation.facts[5]);
  if (!result.key_.valid() || !xpool::ffn::is_valid(result.output_requirement_))
    throw std::invalid_argument("xpool legacy Invocation facts are invalid");
  switch (operation.facts[15]) {
  case 2: {
    auto state = xpool::devkit::fabric_observer::Record::AtnAgentTraceState{};
    state.dp_rank_payload_rows = operation.facts[6];
    state.forward_mode = static_cast<xpool::ffn::ForwardMode>(operation.facts[7]);
    state.dp_row_layout = static_cast<xpool::ffn::DpRowLayout>(operation.facts[8]);
    state.executor_lane_index = operation.facts[9];
    state.executor_lease_sequence = operation.facts[10];
    state.timeline = timeline<xpool::hooks::FabricAtnAgentProtocolEvent::Kind>(operation);
    result.state_ = state;
    break;
  }
  case 3: {
    auto state = xpool::devkit::fabric_observer::Record::CoordinatorTraceState{};
    state.executor_lane_index = operation.facts[9];
    state.executor_lease_sequence = operation.facts[10];
    if (scope.scheduler == LegacyPolicy::Fifo)
      state.ready_ticket = operation.facts[11];
    state.timeline = timeline<xpool::hooks::FabricCoordinatorProtocolEvent::Kind>(operation);
    result.state_ = state;
    break;
  }
  case 4: {
    auto state = xpool::devkit::fabric_observer::Record::FfnAgentTraceState{};
    state.executor_lane_index = operation.facts[9];
    state.executor_lease_sequence = operation.facts[10];
    state.payload_row_capacity = operation.facts[12];
    state.delivery = static_cast<xpool::fabric::DeliveryVariant>(operation.facts[13]);
    if (!xpool::fabric::is_valid(state.delivery))
      throw std::invalid_argument("xpool legacy delivery selection is invalid");
    state.timeline = timeline<xpool::hooks::FabricFfnAgentProtocolEvent::Kind>(operation);
    result.state_ = state;
    break;
  }
  default:
    throw std::invalid_argument("xpool legacy Fabric role is invalid");
  }
  return result;
}

} // namespace xpool::devkit::timeline::detail
