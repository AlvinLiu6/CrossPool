#include <gtest/gtest.h>

#include <xpool/devkit/timeline/fabricevent.hpp>

TEST(FabricObserverTest, RepeatsActualInvocationFactsWithoutLocalTraceIdentity) {
  const auto current = xpool::devkit::timeline::FabricAtnCurrent{
      .submission = {.key = {.instance_index = 0, .invocation_sequence = 7},
                     .layer_ordinal = 2,
                     .payload_rows = 8,
                     .dp_row_layout = xpool::ffn::DpRowLayout::UniformByRank,
                     .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
                     .dp_rank_payload_rows = 8,
                     .forward_mode = xpool::ffn::ForwardMode::Decode}};
  auto event = xpool::devkit::timeline::Event{};
  const auto source = xpool::devkit::timeline::BatchSource{.producer = 2, .clock_domain_id = 3};
  ASSERT_TRUE(xpool::devkit::timeline::make_fabric_atn_event(
      current, source, 1, 0, xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished, 0, event));
  event.event_sequence = 1;
  EXPECT_NO_THROW(event.validate(2, 3));
  EXPECT_EQ(event.invocation_sequence, 7);
  EXPECT_EQ(event.raw_time_ns, 0);
  EXPECT_EQ(event.reserved_zero, 0);
  EXPECT_EQ(event.presence & (1U << 8), 0);
}
