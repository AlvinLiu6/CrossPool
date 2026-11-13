#pragma once

/// \file xpool/devkit/timeline/fabricrecording.cuh
/// \brief Allocation-free Fabric trace preparation and shared recording access.

#include <cuda/atomic>

#include <xpool/devkit/timeline/collection.cuh>
#include <xpool/devkit/timeline/fabricevent.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::timeline {

/// Borrow the process-local binding installed before production launch.
/// An empty result is disabled. The owner retains storage until all production
/// and collection references retire; callers may not save this beyond a Hook.
XPOOL_DEVICE_FN FabricRecordingView fabric_recording_view();

namespace detail {
/// \cond INTERNAL
XPOOL_DEVICE_FN inline bool fabric_installed(FabricRecordingView view) {
  return valid(view.buffer) && view.generation != 0 && view.pe >= 0 && view.state != nullptr &&
         view.instance_count != 0;
}

XPOOL_DEVICE_FN inline void fabric_increment(DeviceCounterState &state) {
  auto exhausted = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.exhausted};
  if (exhausted.load(cuda::memory_order_acquire) != 0)
    return;
  const auto previous = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.value}.fetch_add(
      std::uint64_t{1}, cuda::memory_order_relaxed);
  if (previous == ~std::uint64_t{0})
    exhausted.store(std::uint64_t{1}, cuda::memory_order_release);
}
/// \endcond
} // namespace detail

/// Record one known pre-reservation failure without inventing Source counters.
/// Disabled views are no-ops. Callers select the actual cause; missing Operation
/// is not Context capacity exhaustion, and a conflict is not a new C0 counter.
XPOOL_DEVICE_FN inline void record_fabric_diagnostic(FabricRecordingView view, FabricDiagnostic diagnostic) {
  if (detail::fabric_installed(view))
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{view.state->diagnostics}.fetch_or(
        static_cast<std::uint64_t>(diagnostic), cuda::memory_order_relaxed);
}

/// Shared original record-begin allocation across all roles in this Producer.
XPOOL_DEVICE_FN inline std::uint64_t next_fabric_trace(FabricRecordingView view) {
  detail::fabric_increment(view.state->operations);
  const auto trace = detail::next_device_sequence(view.state->traces);
  if (trace == 0) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{view.state->identity_exhausted}.store(
        std::uint64_t{1}, cuda::memory_order_release);
    detail::source(view.buffer).stop_identity_exhausted();
  } else if (trace > view.record_capacity) {
    detail::fabric_increment(view.state->dropped);
  }
  return trace;
}

/// Snapshot atomic counters on the collection Stream; never read mutable tables.
/// Host inspection is valid only after its enclosing D2H completion succeeds.
XPOOL_DEVICE_FN inline FabricProgress fabric_progress(FabricRecordingView view, bool quiescent) {
  if (!detail::fabric_installed(view))
    return {};
  auto result = FabricProgress{};
  const auto counter = [&result](DeviceCounterState &state) {
    const auto exhausted = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.exhausted}.load(
                               cuda::memory_order_acquire) != 0;
    result.counter_exhausted = result.counter_exhausted || exhausted;
    return exhausted ? ~std::uint64_t{0}
                     : cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{state.value}.load(
                           cuda::memory_order_relaxed);
  };
  result.operations = counter(view.state->operations);
  result.dropped = counter(view.state->dropped);
  result.diagnostics = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{view.state->diagnostics}.load(
      cuda::memory_order_relaxed);
  result.identity_exhausted =
      cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device>{view.state->identity_exhausted}.load(
          cuda::memory_order_acquire) != 0;
  result.exact = quiescent && !result.counter_exhausted;
  return result;
}

/// Prepare exactly once at the real, validated Submission boundary, before
/// both the independent association Hook and ordinary SubmissionPrepared Hook.
/// Only the serialized Instance owner calls this; different Instances share
/// atomic trace allocation. Event capacity, Source stop and listener enablement
/// never allocate a second operation. No call waits or changes production.
XPOOL_DEVICE_FN inline std::uint64_t prepare_fabric_atnagent(FabricRecordingView view,
                                                             const xpool::fabric::Submission &submission) {
  if (!detail::fabric_installed(view))
    return 0;
  if (view.atnagents == nullptr || submission.key.instance_index >= view.instance_count) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return 0;
  }
  auto &current = view.atnagents[submission.key.instance_index];
  current = {};
  if (submission.validate() != xpool::ffn::ResultCode::Ok) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return 0;
  }
  const auto trace = next_fabric_trace(view);
  current.submission = submission;
  current.local_trace_id = trace;
  return trace;
}

/// Read within the synchronous callback for the actual InvocationKey. Key
/// matching prevents a stale consumer from borrowing another operation's trace.
/// This performs bounded atomic reads, no counter increments or allocations.
XPOOL_DEVICE_FN inline FabricRecordingFacts fabric_recording_facts(FabricRecordingView view,
                                                                   xpool::fabric::InvocationKey key) {
  if (!detail::fabric_installed(view))
    return {};
  auto result = FabricRecordingFacts{.status = FabricRecordingStatus::Unavailable,
                                     .buffer = view.buffer,
                                     .generation = view.generation,
                                     .pe = view.pe};
  if (detail::source(view.buffer).is_stopped()) {
    result.status = FabricRecordingStatus::Stopped;
    return result;
  }
  if (view.atnagents == nullptr || key.instance_index >= view.instance_count)
    return result;
  const auto &current = view.atnagents[key.instance_index];
  if (current.local_trace_id == 0 || current.submission.key != key)
    return result;
  result.status = FabricRecordingStatus::Ready;
  result.local_trace_id = current.local_trace_id;
  return result;
}

/// Read raw nanoseconds in the installed Source's Device-globaltimer domain.
/// Zero is valid. Call at this listener's actual boundary, not at trace setup
/// or by looking up a prior event; metadata, not this value, identifies the GPU.
XPOOL_DEVICE_FN inline std::uint64_t fabric_recording_time() { return xpool::utils::time::now(); }

/// Clear after the last local Hook, without resetting any original counter.
/// Only the Instance owner calls this; mismatched keys leave newer facts intact.
XPOOL_DEVICE_FN inline void finish_fabric_atnagent(FabricRecordingView view, xpool::fabric::InvocationKey key) {
  if (detail::fabric_installed(view) && view.atnagents != nullptr && key.instance_index < view.instance_count) {
    auto &current = view.atnagents[key.instance_index];
    if (current.submission.key == key)
      current = {};
  }
}

/// Record one ordinary AtnAgent boundary using the previously prepared trace.
/// One serialized Instance thread calls this. Facts update even if this Event
/// is lost, so later boundaries repeat locally known facts. Source admission
/// owns attempts/full/stale/duplicate counts; this function never recounts them.
XPOOL_DEVICE_FN inline void record_fabric_atn_event(FabricRecordingView view, std::size_t instance_index,
                                                    xpool::hooks::FabricAtnAgentProtocolEvent::Kind kind,
                                                    std::uint64_t time_ns,
                                                    const xpool::fabric::Admission *admission = nullptr,
                                                    const xpool::fabric::OutputCommit *commit = nullptr) {
  if (!detail::fabric_installed(view))
    return;
  if (view.atnagents == nullptr || instance_index >= view.instance_count) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return;
  }
  auto &current = view.atnagents[instance_index];
  if (current.local_trace_id == 0) {
    record_fabric_diagnostic(view, FabricDiagnostic::MissingTrace);
    return;
  }
  using Kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind;
  if (kind == Kind::AdmissionObserved) {
    if (admission == nullptr || admission->validate() != xpool::ffn::ResultCode::Ok ||
        admission->key != current.submission.key) {
      record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
      return;
    }
    current.executor_lane_index = admission->executor_lane_index;
    current.executor_lease_sequence = admission->executor_lease_sequence;
    current.admission_known = true;
  } else if (kind == Kind::OutputCommitObserved) {
    if (commit == nullptr || commit->validate() != xpool::ffn::ResultCode::Ok ||
        commit->key != current.submission.key || !current.admission_known) {
      record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
      return;
    }
    // Production OutputCommit is success-only; failures use FailurePayload.
    current.result = xpool::ffn::ResultCode::Ok;
    current.commit_known = true;
  }
  auto event = Event{};
  if (!make_fabric_atn_event(current, view.buffer.source, view.generation, view.pe, kind, time_ns, event)) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
  } else {
    const auto reserved = try_reserve_device(view.buffer);
    if (reserved.status == ReserveStatus::Reserved &&
        commit_device(view.buffer, reserved.token, event) != CommitStatus::Committed)
      record_fabric_diagnostic(view, FabricDiagnostic::CommitFailure);
  }
  if (kind == Kind::OutputAcknowledgementPublished)
    finish_fabric_atnagent(view, current.submission.key);
}

/// Record one serialized Coordinator transition on its Instance table.
XPOOL_DEVICE_FN inline void record_fabric_coordinator_event(FabricRecordingView view, std::size_t instance,
                                                            xpool::hooks::FabricCoordinatorProtocolEvent::Kind kind,
                                                            const xpool::fabric::Invocation *invocation,
                                                            std::uint64_t ticket, std::size_t lane, std::uint64_t lease,
                                                            std::uint64_t time_ns) {
  if (!detail::fabric_installed(view))
    return;
  if (!view.coordinators || instance >= view.instance_count) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return;
  }
  using Kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind;
  auto &current = view.coordinators[instance];
  if (kind == Kind::Enqueued) {
    current = {};
    if (!invocation || invocation->validate() != xpool::ffn::ResultCode::Ok ||
        invocation->key.instance_index != instance) {
      record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
      return;
    }
    current = {.local_trace_id = next_fabric_trace(view), .invocation = *invocation, .ready_ticket = ticket};
  } else if (kind == Kind::Scheduled) {
    if (lease == 0 || lane >= view.lane_count) {
      record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
      return;
    }
    current.lane_index = lane;
    current.lease_sequence = lease;
    current.scheduled = true;
  }
  auto event = Event{};
  if (make_fabric_coordinator_event(current, view.buffer.source, view.generation, view.pe, view.fifo, kind, time_ns,
                                    event)) {
    const auto reserved = try_reserve_device(view.buffer);
    if (reserved.status == ReserveStatus::Reserved &&
        commit_device(view.buffer, reserved.token, event) != CommitStatus::Committed)
      record_fabric_diagnostic(view, FabricDiagnostic::CommitFailure);
  } else {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
  }
  if (kind == Kind::LaneReleased)
    current = {};
}

/// Record one FfnAgent transition. The original block's elected thread calls
/// this; the Lane remains serialized through its final Completion hook.
XPOOL_DEVICE_FN inline void record_fabric_ffn_event(FabricRecordingView view, std::size_t lane,
                                                    xpool::hooks::FabricFfnAgentProtocolEvent::Kind kind,
                                                    const xpool::fabric::LaneExecution *execution, std::size_t capacity,
                                                    xpool::fabric::DeliveryVariant delivery, std::uint64_t time_ns) {
  if (!detail::fabric_installed(view))
    return;
  if (!view.ffnagents || lane >= view.lane_count) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return;
  }
  using Kind = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;
  auto &current = view.ffnagents[lane];
  if (kind == Kind::LaneExecutionObserved)
    current = {};
  if (!execution || execution->validate() != xpool::ffn::ResultCode::Ok ||
      execution->key.instance_index >= view.instance_count) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return;
  }
  if (kind == Kind::LaneExecutionObserved) {
    if (capacity == 0 || !xpool::fabric::is_valid(delivery)) {
      record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
      return;
    }
    current = {.local_trace_id = next_fabric_trace(view),
               .execution = *execution,
               .payload_row_capacity = capacity,
               .delivery = delivery};
  } else if (current.execution != *execution) {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
    return;
  }
  auto event = Event{};
  if (make_fabric_ffn_event(current, view.buffer.source, view.generation, view.pe, lane, kind, time_ns, event)) {
    const auto reserved = try_reserve_device(view.buffer);
    if (reserved.status == ReserveStatus::Reserved &&
        commit_device(view.buffer, reserved.token, event) != CommitStatus::Committed)
      record_fabric_diagnostic(view, FabricDiagnostic::CommitFailure);
  } else {
    record_fabric_diagnostic(view, FabricDiagnostic::InvalidFacts);
  }
  if (kind == Kind::CompletionPublished)
    current = {};
}

} // namespace xpool::devkit::timeline
