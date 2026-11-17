"""Test-only protocol facts derived from v7 Events; no runtime export format."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import Field


class FabricAtnAgentFactsJson(TypedDict):
    """AtnAgent facts, including the lease acquired after admission."""

    dp_rank_payload_rows: int
    forward_mode: str
    executor_lane_index: int | None
    executor_lease_sequence: int | None


class FabricRandomSchedulerJson(TypedDict):
    """Random scheduling facts without a FIFO ticket."""

    policy: Literal["random"]


class FabricFifoSchedulerJson(TypedDict):
    """FIFO scheduling facts with the invocation's ready ticket."""

    policy: Literal["fifo"]
    ready_ticket: int


class FabricCoordinatorFactsJson(TypedDict):
    """Coordinator scheduling facts, with a nullable pre-schedule lease."""

    executor_lane_index: int | None
    executor_lease_sequence: int | None
    scheduler: FabricRandomSchedulerJson | FabricFifoSchedulerJson


class FabricFfnAgentFactsJson(TypedDict):
    """FfnAgent lease, capacity and delivery facts."""

    executor_lane_index: int
    executor_lease_sequence: int
    payload_row_capacity: int
    delivery: str


class FabricRecordFieldsJson(TypedDict):
    """Scalar invocation fields shared by the native trace alternatives."""

    instance_index: int
    invocation_sequence: int
    layer_ordinal: int
    payload_rows: int
    output_requirement: str


class FabricAtnAgentRecordJson(FabricRecordFieldsJson):
    """One AtnAgent trace and its local event timings."""

    kind: Literal["atnagent"]
    dp_row_layout: str
    facts: FabricAtnAgentFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


class FabricCoordinatorRecordJson(FabricRecordFieldsJson):
    """One Coordinator trace and its local event timings."""

    kind: Literal["coordinator"]
    dp_row_layout: None
    facts: FabricCoordinatorFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


class FabricFfnAgentRecordJson(FabricRecordFieldsJson):
    """One FfnAgent trace and its local event timings."""

    kind: Literal["ffnagent"]
    dp_row_layout: None
    facts: FabricFfnAgentFactsJson
    events_ns: dict[str, int]
    durations_ns: dict[str, int]


type FabricRecordJson = Annotated[
    FabricAtnAgentRecordJson | FabricCoordinatorRecordJson | FabricFfnAgentRecordJson,
    Field(discriminator="kind"),
]


class FabricModelTopologyJson(TypedDict):
    """Attention parallelism attached to one co-indexed Model Plan."""

    atn_tp_size: int
    atn_dp_size: int


class FabricSourceEvidence(TypedDict):
    """One PE's bounded Fabric trace document."""

    generation: str
    pe: int
    atnagent_count: int
    ffnagent_count: int
    model_topologies: list[FabricModelTopologyJson]
    sequence: int
    dropped: int
    records: list[FabricRecordJson]
