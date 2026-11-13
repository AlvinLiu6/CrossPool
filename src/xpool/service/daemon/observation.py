"""Launcher-owned Native identity assignments on the existing participant flow."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal
from uuid import UUID, uuid4

import xpool.native
from xpool.native.devkit import timeline
from xpool.observation import (
    ObservationDeviceRequest,
    ObservationParticipant,
    ObservationPeMember,
    ObservationPreparation,
    process_start_key,
)
from xpool.service.errors import XpoolDaemonError
from xpool.utils.procs import ProcUniqId


class ObservationAdmission:
    """Retain the bounded Run directory as ordinary daemon control-plane state.

    ControlPlane serializes registration and constructs this owner only for
    enabled observation. It never creates a daemon raw Producer or changes C0
    metadata's process-role schema.
    """

    def __init__(self) -> None:
        self.control_budget = timeline.MemoryBudget(1048576)
        self.run_id = uuid4()
        self.directory = timeline.IdentityDirectory(
            self.control_budget, str(self.run_id), process_capacity=128, device_capacity=128
        )

    def participant(
        self,
        process: ProcUniqId,
        request: ObservationDeviceRequest,
        role: Literal["instance", "atnagent", "ffnagent"],
    ) -> ObservationParticipant:
        """Assign or repeat one validated actual process and Device mapping.

        The caller verifies its current registration ownership and policy before
        this cold call. Capacity failures leave previous definitions intact.
        """

        if request.start_key != process_start_key(process):
            raise XpoolDaemonError("conflict", "observation preparation process creation changed")
        try:
            assigned = self.directory.assign_process(
                str(request.host_creation_id), process.pid, request.start_key, xpool.native.RuntimeRole[role.upper()]
            )
            device = self.directory.register_device(
                assigned.process.creation_id, request.logical_device, request.physical_device_uuid, request.raw_epoch
            )
        except timeline.CollectionError as error:
            raise XpoolDaemonError("conflict", f"observation identity admission failed: {error}") from error
        return ObservationParticipant(
            creation_id=UUID(assigned.process.creation_id),
            pid=process.pid,
            runtime_role=role,
            request=request,
            host_producer=assigned.host.producer,
            host_clock=assigned.host.clock_domain_id,
            device_producer=device.producer,
            device_clock=device.clock_domain_id,
        )

    def generation(
        self,
        generation: str,
        members: Iterable[
            tuple[ProcUniqId, ObservationDeviceRequest | None, Literal["instance", "atnagent", "ffnagent"]]
        ],
        pe_owners: Iterable[tuple[int, ProcUniqId]],
    ) -> ObservationPreparation:
        """Freeze assigned Sources for a complete membership before Fabric join or Capture."""

        participants: list[ObservationParticipant] = []
        for process, request, role in members:
            if request is None:
                raise XpoolDaemonError("conflict", "enabled observation requires every participant's preparation")
            participant = self.participant(process, request, role)
            if participant not in participants:
                participants.append(participant)
        try:
            reference = self.directory.register_generation(generation)
        except timeline.CollectionError as error:
            raise XpoolDaemonError("conflict", f"observation Generation admission failed: {error}") from error
        pe_members: list[ObservationPeMember] = []
        for pe, process in pe_owners:
            key = process_start_key(process)
            owner = next(
                (p for p in participants if p.pid == process.pid and p.request.start_key == key),
                None,
            )
            if owner is None:
                raise XpoolDaemonError("conflict", "Fabric PE has no registered observation process owner")
            pe_members.append(
                ObservationPeMember(pe=pe, host_producer=owner.host_producer, device_producer=owner.device_producer)
            )
        return ObservationPreparation(
            run_id=self.run_id,
            generation_ref=reference,
            participants=tuple(participants),
            pe_members=tuple(pe_members),
        )
