"""Routing tensor serialization on the shared Native Writer thread."""

from __future__ import annotations

import functools
import json
from threading import Lock
from typing import Literal, TypedDict

import safetensors.torch
import torch

import xpool.native
from xpool.fabric import FabricParticipantPhase
from xpool.native import RuntimeRole
from xpool.runtime.ffnagent.agent import FfnAgent

runtime_roles = frozenset({RuntimeRole.FFNAGENT})
install_lock = Lock()
installed = False


class RoutingRecord(TypedDict):
    """v7 attachment facts; tensor index is only a storage location."""

    owner: int
    sequence: int
    status: Literal["published"]
    generation: int
    instance_index: int
    invocation_sequence: int
    lane_index: int
    lease_sequence: int
    layer_ordinal: int
    publisher: int
    pe: int
    clock_domain_id: int
    raw_publication_time_ns: int
    shape: list[int]


def encode_routing(
    run_id: str,
    producer_id: int,
    attachments: list[xpool.native.devkit.ffn_routing_observer.RoutingAttachment],
) -> bytes:
    """Serialize Writer-owned CPU tensors using the installed Safetensors library."""

    tensors: dict[str, torch.Tensor] = {}
    records: list[RoutingRecord] = []
    for index, record in enumerate(attachments):
        ids, weights = record.topk_ids, record.topk_weights
        if (
            ids.device.type != "cpu"
            or weights.device.type != "cpu"
            or ids.dtype is not torch.int32
            or weights.dtype is not torch.float32
            or ids.ndim != 2
            or ids.shape != weights.shape
            or not ids.is_contiguous()
            or not weights.is_contiguous()
            or any(size <= 0 for size in ids.shape)
        ):
            raise ValueError(
                "Routing attachment requires contiguous CPU int32/float32 tensors with equal positive 2D shape"
            )
        records.append(
            {
                "owner": producer_id,
                "sequence": record.sequence,
                "status": "published",
                "generation": record.generation,
                "instance_index": record.instance_index,
                "invocation_sequence": record.invocation_sequence,
                "lane_index": record.lane_index,
                "lease_sequence": record.lease_sequence,
                "layer_ordinal": record.layer_ordinal,
                "publisher": producer_id,
                "pe": record.pe,
                "clock_domain_id": record.clock_domain_id,
                "raw_publication_time_ns": record.raw_publication_time_ns,
                "shape": list(ids.shape),
            }
        )
        tensors[f"records.{index}.topk_ids"] = ids
        tensors[f"records.{index}.topk_weights"] = weights
    return safetensors.torch.save(
        tensors,
        metadata={
            "contract_version": "7",
            "run_id": run_id,
            "producer_id": str(producer_id),
            "records": json.dumps(records, separators=(",", ":")),
        },
    )


def install() -> None:
    """Submit immutable tensors once at the actual DRAINED transition."""

    global installed
    with install_lock:
        if installed:
            return
        original = FfnAgent.advance_fabric_lifecycle

        @functools.wraps(original)
        def observed_advance(agent: FfnAgent) -> None:
            previous = agent.participant_report
            previous_phase = previous.phase if previous is not None else None
            original(agent)
            updated = agent.participant_report
            if (
                previous_phase is not FabricParticipantPhase.DRAINED
                and updated is not None
                and updated.phase is FabricParticipantPhase.DRAINED
            ):
                xpool.native.devkit.ffn_routing_observer.collect_drained()

        setattr(FfnAgent, "advance_fabric_lifecycle", observed_advance)
        installed = True
