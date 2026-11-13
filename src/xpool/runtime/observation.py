"""Cold metadata preparation before Fabric join and Instance attachment."""

from __future__ import annotations

import logging
import os

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricPlan, FabricRole, FifoSchedulerPolicy
from xpool.native.devkit import timeline
from xpool.observation import ObservationDeviceRequest, observation_enabled, observation_policy_hash
from xpool.transport import TransportArenaHandle
from xpool.utils.procs import ProcUniqId

__all__ = ["ObservationRuntime", "prepare_observation"]
logger = logging.getLogger(__name__)


def install_generation(metadata: timeline.MetadataRegistry, plan: FabricPlan) -> None:
    """Copy authoritative complete Generation facts into Native typed storage."""

    preparation = plan.observation
    if preparation is None:
        raise ValueError("enabled observation requires Fabric preparation")
    atn_count = sum(placement.role is FabricRole.ATNAGENT for placement in plan.pe_placements)
    metadata.add_generation(
        preparation.generation_ref,
        plan.generation.format(),
        atn_count,
        len(plan.pe_placements) - atn_count,
        isinstance(plan.scheduler, FifoSchedulerPolicy),
    )
    for index, (instance, model) in enumerate(zip(plan.instance_plans, plan.model_plans, strict=True)):
        topology = instance.instance_rank_topology
        metadata.add_model(
            preparation.generation_ref,
            index,
            str(instance.model_id),
            topology.atn_tp_size,
            topology.atn_dp_size,
            model.tp_size,
        )
    for member in preparation.pe_members:
        metadata.add_pe_member(preparation.generation_ref, member.pe, member.host_producer, member.device_producer)


def source_metadata(
    host: timeline.MemoryBudget,
    policy: timeline.CollectionPolicy,
    plan: FabricPlan,
    source: timeline.BatchSource,
    capabilities: timeline.MetadataCapabilities,
) -> timeline.MetadataRegistry:
    """Import assigned Sources before Native serialization, with no Python metadata tree."""

    preparation = plan.observation
    if preparation is None:
        raise ValueError("enabled observation requires assigned process identities")
    limits = timeline.MetadataLimits()
    limits.producers = max(limits.producers, 2 * len(preparation.participants))
    limits.clocks = max(limits.clocks, 2 * len(preparation.participants))
    limits.models_per_generation = max(limits.models_per_generation, len(plan.instance_plans))
    limits.endpoints = max(limits.endpoints, policy.limits.endpoint_capacity)
    metadata = timeline.MetadataRegistry(host, source, policy, capabilities, limits)
    for participant in preparation.participants:
        request = participant.request
        metadata.add_device(
            participant.native_process(preparation.run_id),
            participant.native_device(preparation.run_id),
            request.logical_device,
            request.physical_device_uuid,
            request.raw_epoch,
        )
    install_generation(metadata, plan)
    return metadata


def source_capabilities(policy: timeline.CollectionPolicy) -> timeline.MetadataCapabilities:
    """Leave requested raw providers unavailable until their production Hooks are connected."""

    capabilities = timeline.MetadataCapabilities()
    if not policy.transport_events:
        capabilities.transport = timeline.MetadataCapability.disabled
    if not policy.fabric_events:
        capabilities.fabric = timeline.MetadataCapability.disabled
    if not policy.graph_events:
        capabilities.graph = timeline.MetadataCapability.disabled
    if not policy.routing_events:
        capabilities.routing = timeline.MetadataCapability.disabled
    return capabilities


class ObservationRuntime:
    """Retain prepared metadata and the unique Device IPC pool before Arena creation; no event workers."""

    def __init__(self, plan: FabricPlan, request: ObservationDeviceRequest) -> None:
        preparation = plan.observation
        if preparation is None:
            raise ValueError("enabled observation requires Fabric preparation")
        participant = preparation.participant(ProcUniqId.current())
        if participant.request != request or request.policy_sha256 != observation_policy_hash():
            raise ValueError("observation preparation disagrees with actual registration or policy")
        role = xpool.native.RuntimeRole[participant.runtime_role.upper()]
        if xpool.native.runtime_role() != role:
            raise ValueError("observation preparation disagrees with the initialized runtime role")
        policy = timeline.read_policy()
        if policy is None or not policy.enabled:
            raise ValueError("observation preparation requires enabled bootstrap policy")
        budgets = timeline.prepare_budgets()
        if budgets is None:
            raise ValueError("enabled observation has no admitted Host ledger")
        capabilities = source_capabilities(policy)
        host = participant.native_process(preparation.run_id).host
        device = participant.native_device(preparation.run_id)
        host_metadata = source_metadata(budgets.host, policy, plan, host, capabilities)
        device_capabilities = source_capabilities(policy)
        if policy.fabric_events and role in {xpool.native.RuntimeRole.ATNAGENT, xpool.native.RuntimeRole.FFNAGENT}:
            device_capabilities.fabric = timeline.MetadataCapability.enabled
        if policy.transport_events and role in {xpool.native.RuntimeRole.INSTANCE, xpool.native.RuntimeRole.ATNAGENT}:
            device_capabilities.transport = timeline.MetadataCapability.enabled
        device_metadata = source_metadata(budgets.host, policy, plan, device, device_capabilities)
        host_metadata.validate_preparation(None)
        device_metadata.validate_preparation(request.logical_device)
        self.plan = plan
        self.request = request
        self.host_metadata = host_metadata
        self.device_metadata = device_metadata
        self.participant = participant
        self.pool = (
            timeline.prepare_transport(device, request.logical_device)
            if policy.transport_identity_required and participant.runtime_role in {"instance", "atnagent"}
            else None
        )
        self.generation = preparation.generation_ref
        self.fabric_prepared = False
        if self.pool is not None and participant.runtime_role == "atnagent":
            pe = device_metadata.fabric_pe(self.generation, participant.device_producer)
            count = sum(pe in item.instance_rank_topology.atnagent_indices for item in plan.instance_plans)
            if count > policy.limits.endpoint_capacity:
                raise ValueError("local Transport Arenas exceed prepared observation Endpoint capacity")

    def prepare_fabric(self, projection: xpool.native.fabric.ArenaProjection, pe: int) -> None:
        """Admit the real participant before JOIN_READY; workers start only after Native join."""

        policy = timeline.read_policy()
        if policy is None or not policy.fabric_events or self.fabric_prepared:
            return
        outdir = get_global_config().debug.fabric_observer.outdir
        if outdir is None:
            raise ValueError("Fabric observation requires debug.fabric_observer.outdir")
        xpool.native.devkit.fabric_observer.prepare(self.device_metadata, self.generation, projection, pe, str(outdir))
        self.fabric_prepared = True

    def limit_cleanup(self, deadline: float) -> None:
        """Pass the existing production monotonic deadline without renewing it."""

        timeline.limit_retirement(max(0, int(deadline * 1_000_000_000)))

    def close(self) -> None:
        """Cancel never-activated preparation or join after proven Device retirement.

        Timeout preserves the Native owner and its Host charges. Publication
        failure remains in the bounded report after a successful worker join.
        """

        if not xpool.native.devkit.transport_observer.finish_retirement():
            raise RuntimeError("Transport observation retirement remains unconfirmed; retaining its owner")
        if not self.fabric_prepared:
            return
        observer = xpool.native.devkit.fabric_observer
        if not observer.cancel_preparation() and not observer.finish_retirement():
            raise RuntimeError("Fabric observation retirement remains unconfirmed; retaining its owner")
        self.fabric_prepared = False
        report = observer.export_report()
        if report.error is not None:
            logger.warning("xpool fabric observation retired with export error=%s", report.error)

    def endpoint_owner(self, instance_index: int, rank: int) -> int:
        """Resolve the actual AtnAgent Device Producer from authoritative rank placement."""

        preparation = self.plan.observation
        if preparation is None:
            raise ValueError("Endpoint identity requires prepared membership")
        instance = self.plan.instance_plans[instance_index]
        pe = instance.instance_rank_topology.atnagent_indices[rank]
        return next(member.device_producer for member in preparation.pe_members if member.pe == pe)

    def register_endpoint(
        self,
        handle: TransportArenaHandle,
        instance_index: int,
        rank: int,
        instance_pid: int,
        instance_request: ObservationDeviceRequest | None,
    ) -> None:
        """Validate the prepared pairing and register metadata before publication or CUDA attach.

        The opaque handles remain in the broker descriptor. Metadata retains
        only the Arena suffix and exact registered Device Producers. Equal
        retries are idempotent; changed facts or exhausted tables reject.
        """

        owner = self.endpoint_owner(instance_index, rank)
        handle.validate_observation(required=self.pool is not None, generation=self.generation, owner_producer=owner)
        region = handle.observation_region
        if region is None:
            return
        preparation = self.plan.observation
        if preparation is None:
            raise ValueError("Endpoint identity requires prepared membership")
        participant = next(
            (p for p in preparation.participants if p.pid == instance_pid and p.runtime_role == "instance"), None
        )
        if participant is None or participant.request != instance_request:
            raise ValueError("Endpoint Instance identity differs from actual registration")
        if self.participant.runtime_role == "atnagent" and owner != self.participant.device_producer:
            raise ValueError("Endpoint belongs to a different AtnAgent")
        if self.participant.runtime_role == "instance" and participant != self.participant:
            raise ValueError("Endpoint belongs to a different Instance process")
        for metadata in (self.host_metadata, self.device_metadata):
            metadata.add_endpoint(
                owner,
                region.endpoint_sequence,
                self.generation,
                instance_index,
                rank,
                str(self.plan.instance_plans[instance_index].model_id),
                handle.handle[-16:],
                participant.device_producer,
            )
        self.host_metadata.validate_preparation(None)
        if self.fabric_prepared:
            self.device_metadata.ensure_revision()
        else:
            self.device_metadata.validate_preparation(self.request.logical_device)

        policy = timeline.read_policy()
        if policy is not None and policy.transport_events:
            outdir = get_global_config().debug.transport_observer.outdir
            if outdir is None:
                raise ValueError("Transport observation requires debug.transport_observer.outdir")
            model = self.plan.instance_plans[instance_index].model_id
            site = self.participant.runtime_role
            path = outdir / f"xpool.transport-observer.{os.getpid()}.{site}.{model.uri_encode()}.{rank}.json"
            xpool.native.devkit.transport_observer.prepare_endpoint(
                self.device_metadata,
                region.native(),
                instance_index,
                rank,
                self.request.logical_device,
                site == "atnagent",
                os.getpid(),
                str(model),
                handle.handle[-16:],
                str(path),
            )

    def require_same_preparation(self, plan: FabricPlan, request: ObservationDeviceRequest) -> None:
        """Retry the same immutable plan without reallocating; live replacement needs an explicit retirement path."""

        if self.plan != plan or self.request != request:
            raise ValueError("live observation preparation cannot change Run, Generation or registration")
        if request.policy_sha256 != observation_policy_hash():
            raise ValueError("live observation policy changed after preparation")


def prepare_observation(
    plan: FabricPlan,
    request: ObservationDeviceRequest | None,
    current: ObservationRuntime | None = None,
) -> ObservationRuntime | None:
    """Prepare once on the owning runtime thread; disabled calls allocate or probe nothing."""

    if not observation_enabled():
        if plan.observation is not None or request is not None or current is not None:
            raise ValueError("observation preparation supplied under disabled runtime policy")
        return None
    if request is None:
        raise ValueError("enabled observation requires the process registration request")
    if current is not None:
        current.require_same_preparation(plan, request)
        return current
    return ObservationRuntime(plan, request)
