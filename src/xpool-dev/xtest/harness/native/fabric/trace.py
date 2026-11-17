from __future__ import annotations

from pathlib import Path
from typing import Literal

import safetensors
import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter

import xpool.native
from xpool.config import DebugConfig, FabricObserverDebugConfig, FfnRoutingObserverDebugConfig, GraphObserverDebugConfig
from xpool.native import RuntimeRole
from xtest.harness.native.fabric.protocol import (
    DeliveryVariant,
    FabricAtnAgentTrace,
    FabricCoordinatorTrace,
    FabricFailureEvidence,
    FabricFfnAgentTrace,
    FabricGraphStructureEvidence,
    FabricParticipantReport,
    FabricRoutingEvidence,
    FabricRoutingRecordEvidence,
    FabricTrace,
)
from xtest.harness.support.native.observer import read_fabric_observations
from xtest.harness.support.native.timeline import read_sources


def fabric_debug_options(
    *,
    graph_observer: bool = False,
    routing_observer: bool = False,
) -> xpool.native.debug.Options:
    """Return native debug options for traced Fabric execution."""

    outdir = Path.cwd()
    return DebugConfig(
        fabric_observer=FabricObserverDebugConfig(enable=True, outdir=outdir),
        graph_observer=GraphObserverDebugConfig(enable=graph_observer, outdir=outdir if graph_observer else None),
        ffn_routing_observer=FfnRoutingObserverDebugConfig(
            enable=routing_observer, outdir=outdir if routing_observer else None
        ),
    ).native_options()


class GraphStructureEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["primary", "lane_service"]
    index: int
    binding_site_count: int
    compute_branch_count: int
    delivery_branch_count: int


class RoutingRecordEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instance_index: int
    invocation_sequence: int
    layer_ordinal: int


def fabric_trace_report(
    role: RuntimeRole,
    pe: int,
    outdir: Path,
    failure: xpool.native.fabric.FailurePayload | None,
) -> FabricParticipantReport:
    """Read published v7 artifacts after worker retirement, retaining test facts."""

    sources = read_sources(outdir)
    device = next(
        source
        for source in sources
        if any(p.id == source.metadata.producer_id and p.source == "device" for p in source.metadata.producers)
    )
    assert device.quality is not None
    counters = device.quality.counters
    records: list[FabricTrace] = []
    for source in read_fabric_observations(outdir):
        if source["pe"] != pe:
            continue
        for record in source["records"]:
            common = {
                "instance_index": record["instance_index"],
                "invocation_sequence": record["invocation_sequence"],
                "layer_ordinal": record["layer_ordinal"],
                "payload_rows": record["payload_rows"],
                "executor_lane_index": record["facts"]["executor_lane_index"],
                "executor_lease_sequence": record["facts"]["executor_lease_sequence"],
            }
            events = record["events_ns"]
            if record["kind"] == "atnagent":
                records.append(
                    FabricAtnAgentTrace(
                        **common,
                        forward_mode={
                            "prefill": xpool.native.ffn.ForwardMode.PREFILL,
                            "decode": xpool.native.ffn.ForwardMode.DECODE,
                            "idle": xpool.native.ffn.ForwardMode.IDLE,
                        }[record["facts"]["forward_mode"]],
                        submission_prepared_ns=events.get("submission_prepared", 0),
                        output_acknowledgement_published_ns=events.get("output_acknowledgement_published", 0),
                    )
                )
            elif record["kind"] == "coordinator":
                scheduler = record["facts"]["scheduler"]
                records.append(
                    FabricCoordinatorTrace(
                        **common,
                        ready_ticket=scheduler["ready_ticket"] if scheduler["policy"] == "fifo" else None,
                        enqueued_ns=events.get("enqueued", 0),
                        scheduled_ns=events.get("scheduled", 0),
                        lane_released_ns=events.get("lane_released", 0),
                    )
                )
            else:
                records.append(
                    FabricFfnAgentTrace(
                        **common,
                        payload_row_capacity=record["facts"]["payload_row_capacity"],
                        delivery={
                            "direct_partial": DeliveryVariant.DIRECT_PARTIAL,
                            "single_complete": DeliveryVariant.SINGLE_COMPLETE,
                            "replicated_complete": DeliveryVariant.REPLICATED_COMPLETE,
                        }[record["facts"]["delivery"]],
                        lane_execution_observed_ns=events.get("lane_execution_observed", 0),
                        compute_started_ns=events.get("compute_started", 0),
                        compute_completed_ns=events.get("compute_completed", 0),
                        completion_published_ns=events.get("completion_published", 0),
                    )
                )
    structures = [
        GraphStructureEvidence.model_validate_json(path.read_bytes()) for path in sorted(outdir.rglob("graph/*.json"))
    ]
    primary = sorted((s for s in structures if s.kind == "primary"), key=lambda s: s.index)
    lanes = sorted((s for s in structures if s.kind == "lane_service"), key=lambda s: s.index)
    routing_records: list[FabricRoutingRecordEvidence] = []
    for path in sorted(outdir.rglob("*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as data:
            metadata = data.metadata()
            assert metadata is not None and metadata["contract_version"] == "7"
            attachments = TypeAdapter(list[RoutingRecordEvidence]).validate_json(metadata["records"])
            for index, record in enumerate(attachments):
                ids = data.get_tensor(f"records.{index}.topk_ids")
                weights = data.get_tensor(f"records.{index}.topk_weights")
                routing_records.append(
                    FabricRoutingRecordEvidence(
                        record.instance_index,
                        record.invocation_sequence,
                        record.layer_ordinal,
                        ids.shape[0],
                        ids.shape[1],
                        bytes(ids.view(torch.uint8).flatten().tolist()),
                        bytes(weights.view(torch.uint8).flatten().tolist()),
                    )
                )
    return FabricParticipantReport(
        role=role,
        pe=pe,
        sequence=counters.attempted,
        dropped=counters.dropped_full + counters.post_commit_lost,
        records=tuple(records),
        failure=None
        if failure is None
        else FabricFailureEvidence(
            failure.result_code,
            failure.origin_pe,
            failure.key.instance_index,
            failure.key.invocation_sequence,
            failure.layer_ordinal,
        ),
        graph_structure=FabricGraphStructureEvidence(
            tuple(s.binding_site_count for s in primary),
            tuple(s.compute_branch_count for s in lanes),
            tuple(s.delivery_branch_count for s in lanes),
        )
        if role is RuntimeRole.FFNAGENT
        else None,
        routing=FabricRoutingEvidence(
            counters.attachment_attempted,
            counters.attachment_capacity_exhausted
            + counters.attachment_payload_oversize
            + counters.attachment_publish_failed,
            tuple(routing_records),
        )
        if role is RuntimeRole.FFNAGENT
        else None,
    )
