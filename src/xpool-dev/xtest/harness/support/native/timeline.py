"""Read v7 evidence for assertions; never export runtime compatibility snapshots."""

from __future__ import annotations

import struct
import zlib
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ObservedEvent(BaseModel):
    """Protocol facts needed by native and serving acceptance assertions."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    event_sequence: int
    raw_time_ns: int
    kind: int
    presence: int
    generation: int
    endpoint_owner: int
    endpoint_sequence: int
    operation_sequence: int
    graph_owner: int
    graph_template: int
    graph_execution: int
    data0: int
    data1: int
    data4: int
    instance_index: int
    invocation_sequence: int
    lane_index: int
    lease_sequence: int
    layer_ordinal: int
    payload_rows: int
    dp_rank_payload_rows: int
    payload_row_capacity: int
    ready_ticket: int
    forward_mode: int
    output_requirement: int
    dp_row_layout: int
    delivery: int
    result_code: int
    role: int
    instance_rank: int
    pe: int


class PeMember(BaseModel):
    pe: int
    host_producer: int
    device_producer: int


class ModelTopology(BaseModel):
    model_config = ConfigDict(extra="ignore")
    atn_tp_size: int
    atn_dp_size: int


class Generation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    fabric_generation: str
    atnagent_count: int
    ffnagent_count: int
    scheduler: Literal["fifo", "random"]
    model_topologies: list[ModelTopology]
    pe_members: list[PeMember]


class Process(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pid: int
    runtime_role: Literal["instance", "atnagent", "ffnagent"]


class Producer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    source: Literal["host", "device"]


class QualityCounters(BaseModel):
    model_config = ConfigDict(extra="ignore")
    attempted: int
    dropped_full: int
    post_commit_lost: int
    attachment_attempted: int
    attachment_capacity_exhausted: int
    attachment_payload_oversize: int
    attachment_publish_failed: int


class Quality(BaseModel):
    model_config = ConfigDict(extra="ignore")
    contract_version: Literal[7]
    producer_id: int
    revision: int
    tail: Literal["complete", "pending", "unknown"]
    exact_totals: bool
    counters: QualityCounters


class GraphTemplate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    owner_producer: int
    template_sequence: int
    backend_class: str | None = None


class Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    contract_version: Literal[7]
    run_id: UUID
    producer_id: int
    revision: int
    clock_domain_id: int
    process: Process
    producers: list[Producer]
    graph_templates: list[GraphTemplate]
    generations: list[Generation]


class SourceEvidence(BaseModel):
    """One source and its independently CRC/hash-checked immutable events."""

    root: Path
    metadata: Metadata
    events: list[ObservedEvent]
    quality: Quality | None


EVENT_FIELDS = {
    "event_sequence": (0, "Q"),
    "raw_time_ns": (8, "Q"),
    "kind": (24, "I"),
    "presence": (28, "I"),
    "generation": (32, "Q"),
    "endpoint_owner": (40, "Q"),
    "endpoint_sequence": (48, "Q"),
    "operation_sequence": (56, "Q"),
    "graph_owner": (64, "Q"),
    "graph_template": (72, "Q"),
    "graph_execution": (80, "Q"),
    "data0": (216, "Q"),
    "data1": (224, "Q"),
    "data4": (248, "Q"),
    "instance_index": (88, "Q"),
    "invocation_sequence": (96, "Q"),
    "lane_index": (104, "Q"),
    "lease_sequence": (112, "Q"),
    "layer_ordinal": (144, "Q"),
    "payload_rows": (152, "Q"),
    "dp_rank_payload_rows": (160, "Q"),
    "payload_row_capacity": (168, "Q"),
    "ready_ticket": (176, "Q"),
    "forward_mode": (184, "I"),
    "output_requirement": (188, "I"),
    "dp_row_layout": (192, "I"),
    "delivery": (196, "I"),
    "result_code": (200, "I"),
    "role": (204, "I"),
    "instance_rank": (208, "I"),
    "pe": (212, "i"),
}


def read_sources(outdir: Path) -> list[SourceEvidence]:
    """Verify complete chunks and exact metadata hashes before consuming facts."""

    sources: list[SourceEvidence] = []
    registrations: dict[tuple[Path, int], list[Path]] = {}
    for path in outdir.rglob("metadata.*.*.json"):
        producer = int(path.name.split(".")[1])
        registrations.setdefault((path.parent, producer), []).append(path)
    for directory in outdir.rglob("chunks"):
        for producer_dir in directory.iterdir():
            registrations.setdefault((directory.parent, int(producer_dir.name)), [])
    for (root, producer), revisions in sorted(registrations.items()):
        assert revisions, f"Producer {producer} has chunks but no metadata"
        latest = max(revisions, key=lambda path: int(path.name.split(".")[2]))
        metadata = Metadata.model_validate_json(latest.read_bytes())
        events: dict[int, tuple[bytes, ObservedEvent]] = {}
        for path in sorted((root / "chunks" / str(producer)).glob("*.xpool")):
            data = path.read_bytes()
            assert data[:8] == b"XPOOLC07", path
            assert struct.unpack_from("<HHH", data, 8) == (7, 128, 256), path
            header = bytearray(data[:128])
            checksum = struct.unpack_from("<I", header, 92)[0]
            header[92:96] = bytes(4)
            assert zlib.crc32(header) == checksum, path
            source, _, revision, count = struct.unpack_from("<QQQQ", data, 16)
            assert source == producer and len(data) == 128 + 256 * count, path
            assert zlib.crc32(data[128:]) == struct.unpack_from("<I", data, 88)[0], path
            raw_metadata = (root / f"metadata.{producer}.{revision}.json").read_bytes()
            assert sha256(raw_metadata).digest()[:16] == data[112:128], path
            current = Metadata.model_validate_json(raw_metadata)
            assert current.producer_id == producer and current.run_id.bytes == data[96:112], path
            for offset in range(128, len(data), 256):
                raw = data[offset : offset + 256]
                record = ObservedEvent.model_validate(
                    {name: struct.unpack_from("<" + kind, raw, at)[0] for name, (at, kind) in EVENT_FIELDS.items()}
                )
                assert raw[136:144] == bytes(8) and record.presence & (1 << 8) == 0, path
                if record.event_sequence in events:
                    assert events[record.event_sequence][0] == raw, path
                events[record.event_sequence] = raw, record
        checkpoints = list(root.glob(f"quality.{producer}.*.json"))
        quality = (
            Quality.model_validate_json(max(checkpoints, key=lambda path: int(path.name.split(".")[2])).read_bytes())
            if checkpoints
            else None
        )
        sources.append(
            SourceEvidence(
                root=root,
                metadata=metadata,
                events=[event for _, event in sorted(events.values(), key=lambda pair: pair[1].event_sequence)],
                quality=quality,
            )
        )
    return sources
