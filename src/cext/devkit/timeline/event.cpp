#include <xpool/devkit/timeline/event.hpp>

#include <bit>
#include <optional>
#include <string>
#include <string_view>

#include <xpool/devkit/timeline/contract.hpp>
#include <xpool/devkit/timeline/detail/wire.hpp>
#include <xpool/devkit/timeline/error.hpp>

namespace xpool::devkit::timeline {

void Event::encode(std::span<std::byte, encoded_bytes> destination) const noexcept {
  // Take a value copy before the first write, including when output aliases this.
  const auto event = *this;
  detail::write_integer(destination, 0, event.event_sequence);
  detail::write_integer(destination, 8, event.raw_time_ns);
  detail::write_integer(destination, 16, event.clock_domain_id);
  detail::write_integer(destination, 24, event.kind);
  detail::write_integer(destination, 28, event.presence);
  detail::write_integer(destination, 32, event.generation);
  detail::write_integer(destination, 40, event.endpoint_owner);
  detail::write_integer(destination, 48, event.endpoint_sequence);
  detail::write_integer(destination, 56, event.operation_sequence);
  detail::write_integer(destination, 64, event.graph_owner);
  detail::write_integer(destination, 72, event.graph_template);
  detail::write_integer(destination, 80, event.graph_execution);
  detail::write_integer(destination, 88, event.instance_index);
  detail::write_integer(destination, 96, event.invocation_sequence);
  detail::write_integer(destination, 104, event.lane_index);
  detail::write_integer(destination, 112, event.lease_sequence);
  detail::write_integer(destination, 120, event.attachment_owner);
  detail::write_integer(destination, 128, event.attachment_sequence);
  detail::write_integer(destination, 136, event.local_trace_id);
  detail::write_integer(destination, 144, event.layer_ordinal);
  detail::write_integer(destination, 152, event.payload_rows);
  detail::write_integer(destination, 160, event.dp_rank_payload_rows);
  detail::write_integer(destination, 168, event.payload_row_capacity);
  detail::write_integer(destination, 176, event.ready_ticket);
  detail::write_integer(destination, 184, event.forward_mode);
  detail::write_integer(destination, 188, event.output_requirement);
  detail::write_integer(destination, 192, event.dp_row_layout);
  detail::write_integer(destination, 196, event.delivery);
  detail::write_integer(destination, 200, event.result_code);
  detail::write_integer(destination, 204, event.role);
  detail::write_integer(destination, 208, event.instance_rank);
  detail::write_integer(destination, 212, std::bit_cast<std::uint32_t>(event.pe));
  detail::write_integer(destination, 216, event.data0);
  detail::write_integer(destination, 224, event.data1);
  detail::write_integer(destination, 232, event.data2);
  detail::write_integer(destination, 240, event.data3);
  detail::write_integer(destination, 248, event.data4);
}

Event Event::decode(std::span<const std::byte, encoded_bytes> bytes) noexcept {
  return {
      .event_sequence = detail::read_integer<std::uint64_t>(bytes, 0),
      .raw_time_ns = detail::read_integer<std::uint64_t>(bytes, 8),
      .clock_domain_id = detail::read_integer<std::uint64_t>(bytes, 16),
      .kind = detail::read_integer<std::uint32_t>(bytes, 24),
      .presence = detail::read_integer<std::uint32_t>(bytes, 28),
      .generation = detail::read_integer<std::uint64_t>(bytes, 32),
      .endpoint_owner = detail::read_integer<std::uint64_t>(bytes, 40),
      .endpoint_sequence = detail::read_integer<std::uint64_t>(bytes, 48),
      .operation_sequence = detail::read_integer<std::uint64_t>(bytes, 56),
      .graph_owner = detail::read_integer<std::uint64_t>(bytes, 64),
      .graph_template = detail::read_integer<std::uint64_t>(bytes, 72),
      .graph_execution = detail::read_integer<std::uint64_t>(bytes, 80),
      .instance_index = detail::read_integer<std::uint64_t>(bytes, 88),
      .invocation_sequence = detail::read_integer<std::uint64_t>(bytes, 96),
      .lane_index = detail::read_integer<std::uint64_t>(bytes, 104),
      .lease_sequence = detail::read_integer<std::uint64_t>(bytes, 112),
      .attachment_owner = detail::read_integer<std::uint64_t>(bytes, 120),
      .attachment_sequence = detail::read_integer<std::uint64_t>(bytes, 128),
      .local_trace_id = detail::read_integer<std::uint64_t>(bytes, 136),
      .layer_ordinal = detail::read_integer<std::uint64_t>(bytes, 144),
      .payload_rows = detail::read_integer<std::uint64_t>(bytes, 152),
      .dp_rank_payload_rows = detail::read_integer<std::uint64_t>(bytes, 160),
      .payload_row_capacity = detail::read_integer<std::uint64_t>(bytes, 168),
      .ready_ticket = detail::read_integer<std::uint64_t>(bytes, 176),
      .forward_mode = detail::read_integer<std::uint32_t>(bytes, 184),
      .output_requirement = detail::read_integer<std::uint32_t>(bytes, 188),
      .dp_row_layout = detail::read_integer<std::uint32_t>(bytes, 192),
      .delivery = detail::read_integer<std::uint32_t>(bytes, 196),
      .result_code = detail::read_integer<std::uint32_t>(bytes, 200),
      .role = detail::read_integer<std::uint32_t>(bytes, 204),
      .instance_rank = detail::read_integer<std::uint32_t>(bytes, 208),
      .pe = std::bit_cast<std::int32_t>(detail::read_integer<std::uint32_t>(bytes, 212)),
      .data0 = detail::read_integer<std::uint64_t>(bytes, 216),
      .data1 = detail::read_integer<std::uint64_t>(bytes, 224),
      .data2 = detail::read_integer<std::uint64_t>(bytes, 232),
      .data3 = detail::read_integer<std::uint64_t>(bytes, 240),
      .data4 = detail::read_integer<std::uint64_t>(bytes, 248),
  };
}

void Event::validate(std::uint64_t producer, std::uint64_t source_clock_domain_id) const {
  const auto fail = [&](ParseErrorCode code, std::size_t offset, std::string_view reason) {
    throw ParseError(code, std::string{"xpool "} + std::string{reason}, offset, producer, event_sequence, std::nullopt,
                     std::string{reason});
  };
  const auto known_kind = (kind >= 100 && kind <= 110) || kind == 150 || (kind >= 200 && kind <= 205) ||
                          (kind >= 300 && kind <= 307) || (kind >= 400 && kind <= 408) || kind == 450 ||
                          (kind >= 500 && kind <= 512) || kind == 550 || kind == 600 || kind == 601;
  if (!known_kind)
    fail(ParseErrorCode::UnknownKind, 24, "unknown event kind");
  if (producer == 0 || event_sequence == 0)
    fail(ParseErrorCode::InvalidEvent, 0, "event and source identities must be positive");
  if (clock_domain_id == 0 || clock_domain_id != source_clock_domain_id)
    fail(ParseErrorCode::InvalidEvent, 16, "event clock differs from its source");
  if (role < 1 || role > 5)
    fail(ParseErrorCode::InvalidEvent, 204, "unknown observation role");
  const auto bit = [](unsigned index) { return std::uint32_t{1} << index; };
  const auto present = [&](unsigned index) { return (presence & bit(index)) != 0; };
  if ((presence >> 19) != 0 || (present(2) && !present(1)) ||
      ((present(5) || present(6) || present(17)) && !present(0)) || (present(9) && !present(10)))
    fail(ParseErrorCode::InvalidEvent, 28, "invalid presence mask or group dependencies");
  const auto field = [&](std::uint64_t value, std::uint32_t groups, std::size_t offset, bool positive = false) {
    if ((presence & groups) != 0 ? positive && value == 0 : value != 0)
      fail(ParseErrorCode::InvalidEvent, offset, "absent field is nonzero or present identity is zero");
  };
  field(generation, bit(0), 32, true);
  field(endpoint_owner, bit(1), 40, true);
  field(endpoint_sequence, bit(1), 48, true);
  field(operation_sequence, bit(2), 56, true);
  field(graph_owner, bit(3) | bit(4), 64, true);
  field(graph_template, bit(3), 72, true);
  field(graph_execution, bit(4), 80, true);
  field(instance_index, bit(5), 88);
  field(invocation_sequence, bit(5), 96, true);
  field(lane_index, bit(6), 104);
  field(lease_sequence, bit(6), 112, true);
  field(attachment_owner, bit(7), 120, true);
  field(attachment_sequence, bit(7), 128, true);
  field(local_trace_id, bit(8), 136, true);
  field(layer_ordinal, bit(9) | bit(10), 144);
  field(payload_rows, bit(9) | bit(10), 152);
  field(dp_rank_payload_rows, bit(11), 160);
  field(payload_row_capacity, bit(12), 168);
  field(ready_ticket, bit(13), 176);
  field(forward_mode, bit(9), 184);
  field(output_requirement, bit(9) | bit(10), 188);
  field(dp_row_layout, bit(9), 192);
  field(delivery, bit(14), 196);
  field(result_code, bit(15), 200);
  field(instance_rank, bit(16), 208);
  if (present(17) ? pe < 0 : pe != 0)
    fail(ParseErrorCode::InvalidEvent, 212, "invalid or nonzero absent PE");
  if (present(9) && forward_mode != 1 && forward_mode != 2 && forward_mode != 4)
    fail(ParseErrorCode::InvalidEvent, 184, "unknown forward mode");
  if (present(10) && (output_requirement < 1 || output_requirement > 2))
    fail(ParseErrorCode::InvalidEvent, 188, "unknown output requirement");
  if (present(9) && dp_row_layout > 2)
    fail(ParseErrorCode::InvalidEvent, 192, "unknown DP row layout");
  if (present(14) && (delivery < 1 || delivery > 3))
    fail(ParseErrorCode::InvalidEvent, 196, "unknown delivery");
  if (present(15) && result_code > 3)
    fail(ParseErrorCode::InvalidEvent, 200, "unknown result code");
  if (present(18) && kind != 150)
    fail(ParseErrorCode::InvalidEvent, 28, "raw request is only valid for TransportAnomaly");

  auto required = std::uint32_t{0};
  if (kind == 150) {
    required = bit(0) | bit(1) | bit(2) | bit(8) | bit(16) | bit(18);
    if ((presence & ~(required | bit(15))) != 0 || (role != 1 && role != 2))
      fail(ParseErrorCode::InvalidEvent, 28, "invalid anomalous Transport layout or role");
    const auto mask = transport_invalid_fields(
        static_cast<std::uint32_t>(data1), static_cast<std::uint32_t>(data1 >> 32), static_cast<std::uint32_t>(data2));
    if (mask == 0 || data2 >> 32 != mask)
      fail(ParseErrorCode::InvalidEvent, 232, "raw request invalid mask disagrees with original values");
    const auto own =
        role == 1 ? (data0 >= 100 && data0 <= 102) || (data0 >= 107 && data0 <= 110) : data0 >= 103 && data0 <= 106;
    if (!own || present(15) != (data0 >= 106))
      fail(ParseErrorCode::InvalidEvent, 216, "anomalous boundary or result is not local to this site");
  } else if (kind >= 100 && kind <= 110) {
    required = bit(0) | bit(1) | bit(2) | bit(8) | bit(9) | bit(10) | bit(16);
    if (role != 1 && role != 2)
      fail(ParseErrorCode::InvalidEvent, 204, "Transport role mismatch");
  } else if (kind >= 200 && kind <= 408) {
    required = bit(0) | bit(5) | bit(8) | bit(10) | bit(17);
    const auto expected_role = kind < 300 ? 2U : kind < 400 ? 3U : 4U;
    if (role != expected_role)
      fail(ParseErrorCode::InvalidEvent, 204, "Fabric role mismatch");
    if (role == 2) {
      required |= bit(9) | bit(11);
      if (kind >= 202)
        required |= bit(6);
    } else if (role == 3 && kind >= 301)
      required |= bit(6);
    else if (role == 4)
      required |= bit(6) | bit(12) | bit(14);
  } else if (kind == 450) {
    required = bit(0) | bit(1) | bit(2) | bit(5) | bit(8) | bit(9) | bit(10) | bit(11) | bit(16) | bit(17);
    if (role != 2)
      fail(ParseErrorCode::InvalidEvent, 204, "association role mismatch");
  } else if (kind == 550) {
    required = bit(0) | bit(5) | bit(6) | bit(7) | bit(10) | bit(17);
    if (role != 4 || attachment_owner != producer)
      fail(ParseErrorCode::InvalidEvent, 120, "Routing publisher mismatch");
  }
  if ((presence & required) != required)
    fail(ParseErrorCode::InvalidEvent, 28, "missing mandatory protocol facts");
  if (kind < 500 && kind != 150 && (data0 != 0 || data1 != 0 || data2 != 0 || data3 != 0 || data4 != 0))
    fail(ParseErrorCode::InvalidEvent, 216, "unexpected protocol payload words");
  if (kind >= 500 && kind <= 512) {
    if (data0 < 1 || data0 > 3 || data1 < 1 || data1 > 3 || data4 > 5)
      fail(ParseErrorCode::InvalidEvent, 216, "unknown Graph mode, phase or error");
    const auto activity = kind >= 504 && kind <= 508;
    if (activity ? !present(4) : !present(3) || present(4))
      fail(ParseErrorCode::InvalidEvent, 28, "Graph template or execution identity mismatch");
    if (activity && data0 == 2 && present(3))
      fail(ParseErrorCode::InvalidEvent, 72, "Eager execution cannot carry a template");
    constexpr auto device_boundary = std::uint64_t{1} << 63;
    if (kind == 507 || kind == 508) {
      if (graph_execution <= device_boundary)
        fail(ParseErrorCode::InvalidEvent, 80, "Device Graph execution is outside its identity range");
    } else if (data3 != 0 || (present(4) && graph_execution >= device_boundary))
      fail(ParseErrorCode::InvalidEvent, 80, "Host Graph Context or execution range mismatch");
  } else if (kind == 550) {
    if (data0 < 1 || data0 > 4 || data1 == 0 || data3 != 0 || data4 != 0)
      fail(ParseErrorCode::InvalidEvent, 216, "invalid Routing payload words");
  } else if (kind == 600 || kind == 601) {
    if (presence != 0 || role != 5 || data3 == 0 || data4 == 0)
      fail(ParseErrorCode::InvalidEvent, 28, "invalid sampling identity or role");
    if (kind == 600 ? data0 > data2 || data2 > raw_time_ns : data1 != 0 || data2 < 1 || data2 > 4)
      fail(ParseErrorCode::InvalidEvent, 216, "invalid sampling bracket or failure");
  }
}

} // namespace xpool::devkit::timeline
