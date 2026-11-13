"""Cold process preparation facts carried by existing registration and Fabric flows."""

from __future__ import annotations

import struct
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

import xpool.native
from xpool.config import get_global_config
from xpool.native.devkit import timeline
from xpool.utils.procs import ProcUniqId

__all__ = [
    "ObservationDeviceRequest",
    "ObservationParticipant",
    "ObservationPeMember",
    "ObservationPreparation",
    "host_creation_id",
    "observation_enabled",
    "observation_policy_hash",
    "process_start_key",
]


def process_start_key(process: ProcUniqId) -> int:
    """Preserve exact psutil creation-time bits for retry lookup, separately from UUID identity."""

    return struct.unpack("!Q", struct.pack("!d", process.create_time))[0]


def host_creation_id() -> UUID:
    """Read this Linux boot's identity before any observation registration."""

    return UUID(Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip())


def observation_enabled() -> bool:
    """Resolve collection admission from the installed process configuration."""

    debug = get_global_config().debug
    return debug.timeline.enable or any(
        observer.enable
        for observer in (
            debug.transport_observer,
            debug.fabric_observer,
            debug.graph_observer,
            debug.ffn_routing_observer,
        )
    )


def observation_policy_hash() -> str:
    """Project the same complete debug policy fingerprint installed by bootstrap."""

    return sha256(get_global_config().debug.model_dump_json().encode("utf-8")).hexdigest()


class ObservationDeviceRequest(BaseModel):
    """Actual participant-local CUDA mapping, captured after bootstrap and before registration.

    The daemon verifies process creation, boot, policy and configured physical
    placement. A deployment device index never supplies the local CUDA ordinal.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=5, strict=True, ge=5, le=5)
    host_creation_id: UUID
    start_key: int = Field(strict=True, ge=1, le=2**64 - 1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_device: int = Field(strict=True, ge=0, le=2**31 - 1)
    physical_device_uuid: str = Field(pattern=r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    raw_epoch: int = Field(default=1, strict=True, ge=1, le=1)

    @classmethod
    def current(cls) -> ObservationDeviceRequest | None:
        """Capture the selected primary CUDA context's mapping, or return None when disabled.

        Call after bootstrap selects the actual process device. Enabled calls
        query CUDA properties; disabled calls perform no device or boot probe.
        Context replacement in a live participant is unsupported.
        """

        if not observation_enabled():
            return None
        logical_device = torch.cuda.current_device()
        physical_uuid = str(torch.cuda.get_device_properties(logical_device).uuid)
        physical_uuid = f"GPU-{UUID(physical_uuid.removeprefix('GPU-'))}"
        return cls(
            host_creation_id=host_creation_id(),
            start_key=process_start_key(ProcUniqId.current()),
            policy_sha256=observation_policy_hash(),
            logical_device=logical_device,
            physical_device_uuid=physical_uuid,
        )


class ObservationParticipant(BaseModel):
    """Immutable launcher assignment for one actual Device process creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    creation_id: UUID
    pid: int = Field(strict=True, ge=1, le=2**64 - 1)
    runtime_role: Literal["instance", "atnagent", "ffnagent"]
    request: ObservationDeviceRequest
    host_producer: int = Field(strict=True, ge=1, le=2**64 - 1)
    host_clock: int = Field(strict=True, ge=1, le=2**64 - 1)
    device_producer: int = Field(strict=True, ge=1, le=2**64 - 1)
    device_clock: int = Field(strict=True, ge=1, le=2**64 - 1)

    def native_process(self, run: UUID) -> timeline.RegisteredProcess:
        """Import assigned facts into Native values; this performs no identity allocation or CUDA work."""

        role = xpool.native.RuntimeRole[self.runtime_role.upper()]
        process = timeline.ProcessIdentity(str(self.creation_id), str(self.request.host_creation_id), self.pid, role)
        return timeline.RegisteredProcess(process, timeline.BatchSource(str(run), self.host_producer, self.host_clock))

    def native_device(self, run: UUID) -> timeline.BatchSource:
        """Import this actual Device Producer and raw clock, retaining the launcher's namespace."""

        return timeline.BatchSource(str(run), self.device_producer, self.device_clock)


class ObservationPeMember(BaseModel):
    """One authoritative Generation PE and its same-process Host/Device pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pe: int = Field(strict=True, ge=0, le=2**31 - 1)
    host_producer: int = Field(strict=True, ge=1, le=2**64 - 1)
    device_producer: int = Field(strict=True, ge=1, le=2**64 - 1)


class ObservationPreparation(BaseModel):
    """Bounded B1 preparation envelope on the existing Fabric plan, separate from C0 raw metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=5, strict=True, ge=5, le=5)
    run_id: UUID
    generation_ref: int = Field(strict=True, ge=1, le=2**64 - 1)
    participants: tuple[ObservationParticipant, ...] = Field(min_length=1, max_length=128)
    pe_members: tuple[ObservationPeMember, ...] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def validate_identities(self) -> ObservationPreparation:
        """Reject ambiguous or colliding assignments before any receiver prepares Sources."""

        creations: set[UUID] = set()
        processes: set[tuple[UUID, int, int]] = set()
        producers: set[int] = set()
        clocks: dict[int, tuple[UUID, UUID | None]] = {}
        host_clocks: dict[UUID, int] = {}
        policy = self.participants[0].request.policy_sha256
        for participant in self.participants:
            request = participant.request
            if request.policy_sha256 != policy:
                raise ValueError("observation preparation mixes process policies")
            if (
                request.host_creation_id in host_clocks
                and host_clocks[request.host_creation_id] != participant.host_clock
            ):
                raise ValueError("observation preparation splits one Host boot clock")
            host_clocks[request.host_creation_id] = participant.host_clock
            process = (request.host_creation_id, participant.pid, request.start_key)
            if participant.creation_id in creations or process in processes:
                raise ValueError("observation preparation repeats a process creation")
            creations.add(participant.creation_id)
            processes.add(process)
            for producer in (participant.host_producer, participant.device_producer):
                if producer in producers:
                    raise ValueError("observation preparation repeats a Producer")
                producers.add(producer)
            for clock, owner in (
                (participant.host_clock, (request.host_creation_id, None)),
                (participant.device_clock, (request.host_creation_id, participant.creation_id)),
            ):
                if clock in clocks and clocks[clock] != owner:
                    raise ValueError("observation preparation assigns conflicting raw clock domains")
                clocks[clock] = owner
        seen_pe: set[int] = set()
        seen_producer: set[int] = set()
        for member in self.pe_members:
            if member.pe in seen_pe:
                raise ValueError("observation membership repeats a PE")
            seen_pe.add(member.pe)
            for producer in (member.host_producer, member.device_producer):
                if producer in seen_producer:
                    raise ValueError("observation membership repeats a Producer")
                seen_producer.add(producer)
        agents = tuple(p for p in self.participants if p.runtime_role != "instance")
        atn_count = sum(p.runtime_role == "atnagent" for p in agents)
        if len(self.pe_members) != len(agents) or atn_count == 0 or atn_count == len(agents):
            raise ValueError("observation membership must cover every AtnAgent and FfnAgent")
        by_host = {p.host_producer: p for p in self.participants}
        by_device = {p.device_producer: p for p in self.participants}
        for pe, member in enumerate(self.pe_members):
            host = by_host.get(member.host_producer)
            device = by_device.get(member.device_producer)
            if member.pe != pe or host is None or host != device:
                raise ValueError("observation membership has a gap or mismatched Host/Device owners")
            expected = "atnagent" if pe < atn_count else "ffnagent"
            if host.runtime_role != expected:
                raise ValueError("observation membership disagrees with Fabric process role placement")
        return self

    def participant(self, process: ProcUniqId) -> ObservationParticipant:
        """Find an exact actual process creation; a reused PID cannot consume its predecessor's assignment."""

        key = process_start_key(process)
        boot = host_creation_id()
        for participant in self.participants:
            if (
                participant.pid == process.pid
                and participant.request.start_key == key
                and participant.request.host_creation_id == boot
            ):
                return participant
        raise ValueError("observation preparation has no assignment for this process creation")
