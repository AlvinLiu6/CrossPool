#include <array>

#include <gtest/gtest.h>

#include <xpool/devkit/timeline/transportevent.hpp>

namespace timeline = xpool::devkit::timeline;

namespace {
timeline::TransportRecordingEndpoint endpoint() {
  auto value =
      timeline::TransportRecordingEndpoint{.endpoint_owner = 2, .endpoint_sequence = 1, .generation = 1, .role = 1};
  value.current.operation = {.sequence = 1,
                             .status = xpool::transport::OperationStatus::Available,
                             .endpoint_owner = 2,
                             .endpoint_sequence = 1,
                             .generation = 1};
  value.current.request = {.layer_ordinal = 0,
                           .forward_mode = xpool::ffn::ForwardMode::Decode,
                           .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
                           .dp_row_layout = xpool::ffn::DpRowLayout::None};
  value.current.payload_rows = 2;
  return value;
}
} // namespace

TEST(TransportRecordingTest, PreservesSuccessThenActualShutdownAtClose) {
  auto current = endpoint();
  const auto source = timeline::BatchSource{.producer = 3, .clock_domain_id = 2};
  using Kind = xpool::hooks::TransportProtocolEventKind;
  const auto boundaries = std::array{Kind::ResultObserved, Kind::OutputCopied, Kind::Closed};
  auto sequence = std::uint64_t{0};
  for (const auto kind : boundaries) {
    current.current.observe_result(kind, kind == Kind::Closed ? xpool::ffn::ResultCode::Shutdown
                                                              : xpool::ffn::ResultCode::Ok);
    auto event = timeline::Event{};
    ASSERT_TRUE(timeline::make_transport_event(current, source, kind, 0, event));
    event.event_sequence = ++sequence;
    EXPECT_NO_THROW(event.validate(source.producer, source.clock_domain_id));
    EXPECT_EQ(event.result_code, kind == Kind::Closed ? 1U : 0U);
    EXPECT_EQ(event.raw_time_ns, 0);
    EXPECT_NE(event.presence & static_cast<std::uint32_t>(timeline::Presence::Result), 0U);
  }
}

TEST(TransportRecordingTest, ClosedWithoutPriorResultAndAnomalousRequestRetainActualResult) {
  for (const auto anomalous : {false, true}) {
    for (const auto result : {xpool::ffn::ResultCode::Shutdown, xpool::ffn::ResultCode::ProtocolMismatch}) {
      auto current = endpoint();
      if (anomalous)
        current.current.request.forward_mode = static_cast<xpool::ffn::ForwardMode>(99);
      current.current.observe_result(xpool::hooks::TransportProtocolEventKind::Closed, result);
      auto event = timeline::Event{};
      ASSERT_TRUE(timeline::make_transport_event(current, {.producer = 3, .clock_domain_id = 2},
                                                 xpool::hooks::TransportProtocolEventKind::Closed, 7, event));
      event.event_sequence = 1;
      EXPECT_NO_THROW(event.validate(3, 2));
      EXPECT_EQ(event.kind, anomalous ? 150U : 110U);
      EXPECT_EQ(event.result_code, static_cast<std::uint32_t>(result));
      if (anomalous) {
        EXPECT_EQ(event.data0, 110U);
      }
    }
  }
}

TEST(TransportRecordingTest, OpeningDoesNotInventAResultAndNonResultBoundariesDoNotReplaceIt) {
  auto current = endpoint();
  using Kind = xpool::hooks::TransportProtocolEventKind;
  current.current.observe_result(Kind::RequestStagingStarted, xpool::ffn::ResultCode::Ok);
  EXPECT_FALSE(current.current.result_known);
  auto event = timeline::Event{};
  ASSERT_TRUE(timeline::make_transport_event(current, {.producer = 3, .clock_domain_id = 2},
                                             Kind::RequestStagingStarted, 0, event));
  EXPECT_EQ(event.presence & static_cast<std::uint32_t>(timeline::Presence::Result), 0U);
  current.current.observe_result(Kind::ResultObserved, xpool::ffn::ResultCode::ProtocolMismatch);
  current.current.observe_result(Kind::OutputCopied, xpool::ffn::ResultCode::Ok);
  EXPECT_EQ(current.current.result, xpool::ffn::ResultCode::ProtocolMismatch);
}
