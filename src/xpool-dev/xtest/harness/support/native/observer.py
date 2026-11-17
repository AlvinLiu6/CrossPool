"""Assertions for structured native observer evidence in integration tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from xtest.harness.support.native.timeline import ObservedEvent, read_sources
from xtest.harness.support.native.tracefacts import (
    FabricAtnAgentRecordJson,
    FabricCoordinatorRecordJson,
    FabricFfnAgentRecordJson,
    FabricRecordFieldsJson,
    FabricRecordJson,
    FabricSourceEvidence,
)


def assert_transport_observations(
    outdir: Path,
    *,
    expected_count: int,
    site: Literal["instance", "atnagent"] | None = None,
) -> None:
    """Require complete, successful, gap-free Transport mailbox traces."""

    groups: dict[tuple[int, int, int, int], dict[int, ObservedEvent]] = {}
    for source in read_sources(outdir):
        for event in source.events:
            if not 100 <= event.kind <= 110:
                continue
            if site is not None and event.role != (1 if site == "instance" else 2):
                continue
            key = (event.endpoint_owner, event.endpoint_sequence, event.role, event.instance_rank)
            groups.setdefault(key, {})[event.operation_sequence * 1000 + event.kind] = event
    assert len(groups) == expected_count
    for events in groups.values():
        operations: dict[int, set[int]] = {}
        for event in events.values():
            operations.setdefault(event.operation_sequence, set()).add(event.kind)
            if event.presence & (1 << 15):
                assert event.result_code == 0
        assert operations
        for kinds in operations.values():
            assert 109 in kinds if min(kinds) <= 102 else 106 in kinds


def assert_fabric_observations(
    outdir: Path,
    *,
    atnagent_count: int,
    ffnagent_count: int,
    expected_topologies: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Require complete AtnAgent, Coordinator, and FfnAgent trace alternatives."""

    snapshots = read_fabric_observations(outdir)
    expected_count = atnagent_count + ffnagent_count
    assert len(snapshots) == expected_count
    assert len({snapshot["generation"] for snapshot in snapshots}) == 1
    assert {snapshot["pe"] for snapshot in snapshots} == set(range(expected_count))
    assert all(snapshot["atnagent_count"] == atnagent_count for snapshot in snapshots)
    assert all(snapshot["ffnagent_count"] == ffnagent_count for snapshot in snapshots)
    if expected_topologies is not None:
        expected = [
            {"atn_tp_size": atn_tp_size, "atn_dp_size": atn_dp_size} for atn_tp_size, atn_dp_size in expected_topologies
        ]
        assert all(snapshot["model_topologies"] == expected for snapshot in snapshots)

    atnagent_records: list[tuple[int, FabricAtnAgentRecordJson]] = []
    coordinator_records: list[tuple[int, FabricCoordinatorRecordJson]] = []
    ffnagent_records: list[tuple[int, FabricFfnAgentRecordJson]] = []
    for snapshot in snapshots:
        records = snapshot["records"]
        assert snapshot["sequence"] > 0
        assert snapshot["dropped"] == 0
        assert records
        pe = snapshot["pe"]
        for record in records:
            match record["kind"]:
                case "atnagent":
                    atnagent_records.append((pe, record))
                case "coordinator":
                    coordinator_records.append((pe, record))
                case "ffnagent":
                    ffnagent_records.append((pe, record))
            assert record["output_requirement"] in {"per_rank_complete", "group_sum_complete"}
            assert record["dp_row_layout"] in {None, "none", "uniform_by_rank", "packed_by_rank"}

    assert_atnagent_records(atnagent_records, atnagent_count=atnagent_count)
    assert_coordinator_records(coordinator_records, coordinator_pe=atnagent_count)
    assert_ffnagent_records(
        ffnagent_records,
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
    )
    assert_invocation_correspondence(
        {"atnagent": atnagent_records, "coordinator": coordinator_records, "ffnagent": ffnagent_records},
        atnagent_count=atnagent_count,
        ffnagent_count=ffnagent_count,
    )


def assert_two_model_executor_overlap(outdir: Path) -> None:
    """Require model-zero and model-one Active intervals to overlap on distinct Executors."""

    snapshots = read_fabric_observations(outdir)
    coordinator_records = [
        record for snapshot in snapshots for record in snapshot["records"] if record["kind"] == "coordinator"
    ]
    intervals_by_model: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for record in coordinator_records:
        instance_index = record["instance_index"]
        if instance_index not in intervals_by_model:
            continue
        facts = record["facts"]
        events = record["events_ns"]
        executor_lane_index = facts["executor_lane_index"]
        scheduled = events["scheduled"]
        released = events["lane_released"]
        if scheduled > 0 and released >= scheduled:
            assert executor_lane_index is not None
            intervals_by_model[instance_index].append((executor_lane_index, scheduled, released))

    overlap_durations = [
        max(0, min(left_end, right_end) - max(left_start, right_start))
        for left_executor, left_start, left_end in intervals_by_model[0]
        for right_executor, right_start, right_end in intervals_by_model[1]
        if left_executor != right_executor
    ]
    maximum_overlap_ns = max(overlap_durations, default=0)
    executors_by_model = {
        instance_index: sorted({executor for executor, _, _ in intervals})
        for instance_index, intervals in intervals_by_model.items()
    }
    assert maximum_overlap_ns > 0, (
        "two-model E2E did not observe overlapping Active intervals on distinct Executors: "
        f"interval_counts={{0: {len(intervals_by_model[0])}, 1: {len(intervals_by_model[1])}}}, "
        f"executors={executors_by_model}, candidate_pairs={len(overlap_durations)}, "
        f"maximum_overlap_ns={maximum_overlap_ns}"
    )


def assert_dp_attention_paths(outdir: Path) -> None:
    """Require active/Idle ranks and both accepted DP layout/output paths."""

    records = [
        record
        for snapshot in read_fabric_observations(outdir)
        for record in snapshot["records"]
        if record["kind"] == "atnagent"
    ]
    assert records
    facts = [record["facts"] for record in records]
    assert {"prefill", "decode", "idle"} <= {fact["forward_mode"] for fact in facts}
    assert {"packed_by_rank", "uniform_by_rank"} <= {record["dp_row_layout"] for record in records}
    assert {"per_rank_complete", "group_sum_complete"} <= {record["output_requirement"] for record in records}


def read_fabric_observations(outdir: Path) -> list[FabricSourceEvidence]:
    """Read every per-PE Fabric observer document in deterministic PE order."""

    snapshots: list[FabricSourceEvidence] = []
    for source in read_sources(outdir):
        own = next(p for p in source.metadata.producers if p.id == source.metadata.producer_id)
        if own.source != "device" or source.metadata.process.runtime_role == "instance":
            continue
        events = [event for event in source.events if 200 <= event.kind <= 408]
        generation = next(
            g for g in source.metadata.generations if any(member.device_producer == own.id for member in g.pe_members)
        )
        pe = next(member.pe for member in generation.pe_members if member.device_producer == own.id)
        assert source.quality is not None
        counters = source.quality.counters
        grouped: dict[tuple[int, int, int], list[ObservedEvent]] = {}
        for event in events:
            grouped.setdefault((event.role, event.instance_index, event.invocation_sequence), []).append(event)
        records: list[FabricRecordJson] = []
        for group in grouped.values():
            event = max(group, key=lambda e: e.presence.bit_count())
            times: dict[str, int] = {}
            names = {
                200: "submission_prepared",
                201: "submission_published",
                202: "admission_observed",
                203: "input_ready_published",
                204: "output_commit_observed",
                205: "output_acknowledgement_published",
                300: "enqueued",
                301: "scheduled",
                302: "admission_published",
                303: "lane_execution_published",
                304: "ffnagent_completions_observed",
                305: "output_commit_published",
                306: "output_acknowledgements_observed",
                307: "lane_released",
                400: "lane_execution_observed",
                401: "input_ready_observed",
                402: "routing_metadata_published",
                403: "routing_metadata_observed",
                404: "compute_started",
                405: "compute_completed",
                406: "partial_ready_published",
                407: "peer_partials_ready_observed",
                408: "completion_published",
            }
            for boundary in group:
                name = names[boundary.kind]
                if name in times:
                    assert times[name] == boundary.raw_time_ns
                times[name] = boundary.raw_time_ns
            common: FabricRecordFieldsJson = {
                "instance_index": event.instance_index,
                "invocation_sequence": event.invocation_sequence,
                "layer_ordinal": event.layer_ordinal,
                "payload_rows": event.payload_rows,
                "output_requirement": {1: "per_rank_complete", 2: "group_sum_complete"}[event.output_requirement],
            }
            lane = event.lane_index if event.presence & (1 << 6) else None
            lease = event.lease_sequence if event.presence & (1 << 6) else None
            if event.role == 2:
                records.append(
                    {
                        **common,
                        "kind": "atnagent",
                        "dp_row_layout": {0: "none", 1: "uniform_by_rank", 2: "packed_by_rank"}[event.dp_row_layout],
                        "facts": {
                            "dp_rank_payload_rows": event.dp_rank_payload_rows,
                            "forward_mode": {1: "prefill", 2: "decode", 4: "idle"}[event.forward_mode],
                            "executor_lane_index": lane,
                            "executor_lease_sequence": lease,
                        },
                        "events_ns": times,
                        "durations_ns": {},
                    }
                )
            elif event.role == 3:
                records.append(
                    {
                        **common,
                        "kind": "coordinator",
                        "dp_row_layout": None,
                        "facts": {
                            "executor_lane_index": lane,
                            "executor_lease_sequence": lease,
                            "scheduler": {"policy": "fifo", "ready_ticket": event.ready_ticket}
                            if generation.scheduler == "fifo"
                            else {"policy": "random"},
                        },
                        "events_ns": times,
                        "durations_ns": {},
                    }
                )
            else:
                assert lane is not None and lease is not None
                records.append(
                    {
                        **common,
                        "kind": "ffnagent",
                        "dp_row_layout": None,
                        "facts": {
                            "executor_lane_index": lane,
                            "executor_lease_sequence": lease,
                            "payload_row_capacity": event.payload_row_capacity,
                            "delivery": {1: "direct_partial", 2: "single_complete", 3: "replicated_complete"}[
                                event.delivery
                            ],
                        },
                        "events_ns": times,
                        "durations_ns": {},
                    }
                )
        snapshots.append(
            {
                "generation": generation.fabric_generation,
                "pe": pe,
                "atnagent_count": generation.atnagent_count,
                "ffnagent_count": generation.ffnagent_count,
                "model_topologies": [
                    {"atn_tp_size": model.atn_tp_size, "atn_dp_size": model.atn_dp_size}
                    for model in generation.model_topologies
                ],
                "sequence": counters.attempted,
                "dropped": counters.dropped_full + counters.post_commit_lost,
                "records": records,
            }
        )
    return sorted(snapshots, key=lambda source: source["pe"])


def assert_atnagent_records(records: list[tuple[int, FabricAtnAgentRecordJson]], *, atnagent_count: int) -> None:
    """Validate AtnAgent mode-sensitive publication and result events."""

    assert records
    for pe, record in records:
        assert pe < atnagent_count
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["forward_mode"] in {"prefill", "decode", "idle"}
        assert record["payload_rows"] > 0
        assert facts["dp_rank_payload_rows"] >= 0
        lease_sequence = facts["executor_lease_sequence"]
        assert lease_sequence is not None and lease_sequence > 0
        require_positive_events(
            events,
            (
                "submission_prepared",
                "submission_published",
                "admission_observed",
                "output_commit_observed",
                "output_acknowledgement_published",
            ),
        )


def assert_coordinator_records(records: list[tuple[int, FabricCoordinatorRecordJson]], *, coordinator_pe: int) -> None:
    """Validate complete Coordinator scheduling and publication events."""

    assert records
    for pe, record in records:
        assert pe == coordinator_pe
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["scheduler"] is not None
        lease_sequence = facts["executor_lease_sequence"]
        assert lease_sequence is not None and lease_sequence > 0
        require_positive_events(
            events,
            (
                "enqueued",
                "scheduled",
                "admission_published",
                "lane_execution_published",
                "ffnagent_completions_observed",
                "output_commit_published",
                "output_acknowledgements_observed",
                "lane_released",
            ),
        )


def assert_ffnagent_records(
    records: list[tuple[int, FabricFfnAgentRecordJson]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Validate every selected physical FfnAgent contribution."""

    assert records
    for pe, record in records:
        assert atnagent_count <= pe < atnagent_count + ffnagent_count
        facts = record["facts"]
        events = record["events_ns"]
        assert facts["executor_lease_sequence"] > 0
        assert facts["payload_row_capacity"] >= record["payload_rows"]
        assert facts["delivery"] in {"direct_partial", "single_complete", "replicated_complete"}
        require_positive_events(
            events,
            (
                "lane_execution_observed",
                "input_ready_observed",
                "compute_started",
                "compute_completed",
                "completion_published",
            ),
        )


def assert_invocation_correspondence(
    records_by_kind: Mapping[str, Sequence[tuple[int, FabricRecordJson]]],
    *,
    atnagent_count: int,
    ffnagent_count: int,
) -> None:
    """Match each invocation across AtnAgent, Coordinator, and selected FfnAgent records."""

    grouped: dict[str, dict[tuple[int, int, int, int | None], list[tuple[int, FabricRecordJson]]]] = {}
    for kind, records in records_by_kind.items():
        by_key: dict[tuple[int, int, int, int | None], list[tuple[int, FabricRecordJson]]] = {}
        for pe, record in records:
            by_key.setdefault(invocation_key(record), []).append((pe, record))
        grouped[kind] = by_key
    assert grouped["atnagent"].keys() == grouped["coordinator"].keys() == grouped["ffnagent"].keys()
    assert all(len(records) == atnagent_count for records in grouped["atnagent"].values())
    assert all(
        any(record["events_ns"]["input_ready_published"] > 0 for _, record in records)
        for records in grouped["atnagent"].values()
    )
    assert all(len(records) == 1 for records in grouped["coordinator"].values())
    assert all(1 <= len(records) <= ffnagent_count for records in grouped["ffnagent"].values())


def invocation_key(record: FabricRecordJson) -> tuple[int, int, int, int | None]:
    """Return the semantic identity shared by all trace alternatives."""

    facts = record["facts"]
    return (
        record["instance_index"],
        record["invocation_sequence"],
        record["layer_ordinal"],
        facts["executor_lane_index"],
    )


def require_positive_events(events: dict[str, int], names: tuple[str, ...]) -> None:
    """Require every named event timestamp to be present."""

    assert all(events[name] > 0 for name in names)
