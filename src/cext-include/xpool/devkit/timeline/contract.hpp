#pragma once

/// \file xpool/devkit/timeline/contract.hpp
/// \brief C0 v5 candidate vocabulary, independent of production enum ordinals.

#include <cstdint>
#include <string_view>

#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Single supported artifact and observation IPC contract version.
inline constexpr std::uint32_t kContractVersion = 5;
/// Eight literal raw chunk signature bytes, without a terminating NUL.
inline constexpr auto kChunkMagic = std::string_view{"XPOOLC05"};

/// Explicit wire identifiers; map production Hook enums independently.
enum class EventKind : std::uint32_t {
  RequestStagingStarted = 100,
  RequestStagingCompleted = 101,
  RequestPublished = 102,
  RequestObserved = 103,
  ExecutionStarted = 104,
  ExecutionCompleted = 105,
  ResultPublished = 106,
  ResultObserved = 107,
  OutputCopied = 108,
  ResultAcknowledged = 109,
  Closed = 110,
  TransportAnomaly = 150,
  SubmissionPrepared = 200,
  SubmissionPublished = 201,
  AdmissionObserved = 202,
  InputReadyPublished = 203,
  OutputCommitObserved = 204,
  OutputAcknowledgementPublished = 205,
  Enqueued = 300,
  Scheduled = 301,
  AdmissionPublished = 302,
  LaneExecutionPublished = 303,
  FfnAgentCompletionsObserved = 304,
  OutputCommitPublished = 305,
  OutputAcknowledgementsObserved = 306,
  LaneReleased = 307,
  LaneExecutionObserved = 400,
  InputReadyObserved = 401,
  RoutingMetadataPublished = 402,
  RoutingMetadataObserved = 403,
  ComputeStarted = 404,
  ComputeCompleted = 405,
  PartialReadyPublished = 406,
  PeerPartialsReadyObserved = 407,
  CompletionPublished = 408,
  TransportFabricAssociated = 450,
  GraphCaptureBegin = 500,
  GraphCaptureEnd = 501,
  GraphCaptureError = 502,
  GraphTemplateInstalled = 503,
  GraphHostCallBegin = 504,
  GraphHostCallEnd = 505,
  GraphHostCallError = 506,
  GraphDeviceActivityBegin = 507,
  GraphDeviceActivityEnd = 508,
  GraphTemplateRetired = 509,
  GraphInstallBegin = 510,
  GraphInstallEnd = 511,
  GraphInstallError = 512,
  RoutingPublished = 550,
  CalibrationSampleSucceeded = 600,
  CalibrationSampleFailed = 601,
};

/// Observation site, distinct from process runtime role.
enum class EventRole : std::uint32_t { Instance = 1, AtnAgent = 2, Coordinator = 3, FfnAgent = 4, Collector = 5 };

/// Known-fact field groups; zero-valued present fields remain observed facts.
enum class Presence : std::uint32_t {
  Generation = 1U << 0,
  Endpoint = 1U << 1,
  Operation = 1U << 2,
  GraphTemplate = 1U << 3,
  GraphExecution = 1U << 4,
  Invocation = 1U << 5,
  Lease = 1U << 6,
  Attachment = 1U << 7,
  LocalTrace = 1U << 8,
  RequestFacts = 1U << 9,
  InvocationFacts = 1U << 10,
  DpRows = 1U << 11,
  RowCapacity = 1U << 12,
  ReadyTicket = 1U << 13,
  Delivery = 1U << 14,
  Result = 1U << 15,
  InstanceRank = 1U << 16,
  Pe = 1U << 17,
  RawRequest = 1U << 18,
};

/// Exact invalid-enum mask for raw Transport request fields. Bits 0/1/2
/// identify forward mode, output requirement and DP layout respectively.
XPOOL_HOST_DEVICE_FN constexpr std::uint32_t transport_invalid_fields(std::uint32_t forward, std::uint32_t output,
                                                                      std::uint32_t layout) noexcept {
  return (forward != 1 && forward != 2 && forward != 4 ? 1U : 0U) | (output != 1 && output != 2 ? 2U : 0U) |
         (layout > 2 ? 4U : 0U);
}

} // namespace xpool::devkit::timeline
