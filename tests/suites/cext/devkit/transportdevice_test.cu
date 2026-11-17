#include <array>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/devkit/timeline/transportrecording.cuh>

namespace timeline = xpool::devkit::timeline;

namespace {
struct RecordingState {
  timeline::Event events[8]{};
  timeline::SlotControl slots[8]{};
  timeline::SlotPoolState pool{};
  timeline::DeviceSourceState source{};
  timeline::TransportRecordingEndpoint endpoint{};
};

XPOOL_KERNEL_FN void record_boundaries(RecordingState *state) {
  const auto buffer = timeline::BufferView{.source = {.producer = 3, .clock_domain_id = 2},
                                           .events = state->events,
                                           .slots = state->slots,
                                           .pool_state = &state->pool,
                                           .source_state = &state->source,
                                           .capacity = 8};
  const auto view = timeline::TransportRecordingView{buffer, &state->endpoint, 1};
  const auto operation = xpool::transport::OperationFact{.sequence = 1,
                                                         .status = xpool::transport::OperationStatus::Available,
                                                         .endpoint_owner = 2,
                                                         .endpoint_sequence = 1,
                                                         .generation = 1};
  const auto request =
      xpool::transport::RequestMetadata{.layer_ordinal = 0,
                                        .forward_mode = xpool::ffn::ForwardMode::Decode,
                                        .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
                                        .dp_row_layout = xpool::ffn::DpRowLayout::None};
  using Kind = xpool::hooks::TransportProtocolEventKind;
  constexpr Kind boundaries[] = {Kind::RequestStagingStarted, Kind::ResultObserved, Kind::OutputCopied, Kind::Closed};
  for (const auto boundary : boundaries) {
    timeline::record_transport_event(
        view, state->endpoint, boundary, operation, &request, 2,
        boundary == Kind::Closed ? xpool::ffn::ResultCode::Shutdown : xpool::ffn::ResultCode::Ok, 0);
  }
}

class TransportDeviceRecordingTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0)
      GTEST_SKIP() << "device is not available: " << cudaGetErrorString(status);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void **>(&state_), sizeof(RecordingState)), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    *state_ = RecordingState{};
    state_->endpoint = {.endpoint_owner = 2, .endpoint_sequence = 1, .generation = 1, .role = 1};
  }
  void TearDown() override {
    if (state_) {
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      EXPECT_EQ(cudaFree(state_), cudaSuccess);
    }
    if (stream_)
      EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }
  RecordingState *state_ = nullptr;
  cudaStream_t stream_ = nullptr;
};
} // namespace

TEST_F(TransportDeviceRecordingTest, CommitsActualShutdownWithoutChangingEarlierSuccess) {
  record_boundaries<<<1, 1, 0, stream_>>>(state_);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  auto kinds = std::array<bool, 4>{};
  for (const auto &event : state_->events) {
    if (event.event_sequence == 0)
      continue;
    EXPECT_NO_THROW(event.validate(3, 2));
    EXPECT_EQ(event.raw_time_ns, 0);
    switch (event.kind) {
    case 100:
      kinds[0] = true;
      EXPECT_EQ(event.presence & (1U << 15), 0U);
      break;
    case 107:
      kinds[1] = true;
      EXPECT_EQ(event.result_code, 0U);
      break;
    case 108:
      kinds[2] = true;
      EXPECT_EQ(event.result_code, 0U);
      break;
    case 110:
      kinds[3] = true;
      EXPECT_EQ(event.result_code, 1U);
      break;
    default:
      FAIL() << "unexpected Transport boundary " << event.kind;
    }
  }
  EXPECT_EQ(kinds, (std::array{true, true, true, true}));
  EXPECT_EQ(state_->endpoint.diagnostics, 0U);
}
