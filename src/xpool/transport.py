"""Typed host lifecycle contracts for native CUDA IPC transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

import xpool.native
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement

__all__ = [
    "TRANSPORT_ARENA_HANDLE_HEX_LENGTH",
    "FfnRequestMetadata",
    "TransportArenaHandle",
    "TransportObservationRegion",
]

TRANSPORT_ARENA_HANDLE_HEX_LENGTH = int(xpool.native.transport.ARENA_HANDLE_HEX_LENGTH)
"""Character count of a lowercase hexadecimal CUDA IPC memory handle."""


@dataclass(frozen=True, slots=True)
class FfnRequestMetadata:
    """Transport wire metadata for one public FFN shim call.

    Attributes:
        layer_ordinal: Zero-based FFN position in the Fabric workload.
        forward_mode: Prefill, decode, or idle ABI value.
        output_requirement: Mathematical output requirement.
        dp_row_layout: Physical data-parallel row layout selected by the producer.
    """

    layer_ordinal: int
    forward_mode: ForwardMode
    output_requirement: OutputRequirement
    dp_row_layout: DpRowLayout

    def __post_init__(self) -> None:
        """Reject untyped request metadata before dispatcher entry."""

        if isinstance(self.layer_ordinal, bool) or not isinstance(self.layer_ordinal, int) or self.layer_ordinal < 0:
            raise ValueError("xpool FFN layer ordinal must be a non-negative integer")
        if not isinstance(self.forward_mode, ForwardMode):
            raise TypeError("xpool FFN forward mode must be ForwardMode")
        if not isinstance(self.output_requirement, OutputRequirement):
            raise TypeError("xpool FFN output requirement must be OutputRequirement")
        if not isinstance(self.dp_row_layout, DpRowLayout):
            raise TypeError("xpool FFN DP row layout must be DpRowLayout")


class TransportObservationRegion(BaseModel):
    """Immutable C0 observation descriptor carried beside an Arena handle.

    This value contains only control-plane facts. It owns no CUDA mapping and
    never enters raw timeline metadata. Successful parsing proves field shape;
    preparation must separately check the actual owner, Generation and policy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(strict=True, ge=5, le=5, description="Required C0 descriptor version.")
    owner_producer: int = Field(strict=True, ge=1, le=2**64 - 1, description="AtnAgent Device Producer.")
    endpoint_sequence: int = Field(strict=True, ge=1, le=2**64 - 1, description="Fresh Producer-local Endpoint.")
    generation: int = Field(strict=True, ge=1, le=2**64 - 1, description="Generation reference at allocation.")
    allocation_bytes: int = Field(
        strict=True, ge=8, le=2**64 - 1, multiple_of=8, description="Allocation extent including alignment."
    )
    handle: str = Field(
        strict=True,
        min_length=TRANSPORT_ARENA_HANDLE_HEX_LENGTH,
        max_length=TRANSPORT_ARENA_HANDLE_HEX_LENGTH,
        pattern=r"^[0-9a-f]+$",
        description="Canonical CUDA IPC handle; Native checks its exact allocation extent before attachment.",
    )

    def native(self) -> xpool.native.transport.ObservationRegion:
        """Construct the immutable Native value; this performs no CUDA calls."""

        return xpool.native.transport.ObservationRegion(
            self.version,
            self.owner_producer,
            self.endpoint_sequence,
            self.generation,
            self.allocation_bytes,
            self.handle,
        )


@dataclass(frozen=True, slots=True)
class TransportArenaHandle:
    """CUDA IPC transport arena handle brokered by the daemon.

    Attributes:
        handle: Lowercase hex CUDA IPC handle used as control-plane identity.
        observation_region: Optional C0 descriptor, present only when immutable
            preparation requires shared Transport identity.
    """

    __pydantic_config__ = ConfigDict(extra="forbid")

    handle: Annotated[str, Field(strict=True)]
    observation_region: TransportObservationRegion | None = None

    def __post_init__(self) -> None:
        """Validate the handle at Python transport boundaries."""

        if len(self.handle) != TRANSPORT_ARENA_HANDLE_HEX_LENGTH:
            raise ValueError(
                f"transport arena handle must contain {TRANSPORT_ARENA_HANDLE_HEX_LENGTH} lowercase hex characters"
            )
        if any(character not in "0123456789abcdef" for character in self.handle):
            raise ValueError("transport arena handle must contain only lowercase hex characters")
        if self.observation_region is not None and not isinstance(self.observation_region, TransportObservationRegion):
            raise TypeError("transport observation region must be a validated descriptor")

    def validate_observation(self, *, required: bool, generation: int, owner_producer: int) -> None:
        """Check the descriptor against actual preparation before publishing or attaching.

        Args:
            required: Frozen shared-identity policy for both participants.
            generation: Registered Generation reference of this Arena pairing.
            owner_producer: Actual remote AtnAgent Device Producer, not the importer.

        Raises:
            ValueError: If policy, owner or Generation differs. This Host check
                neither opens CUDA handles nor proves peer detach or liveness.
        """

        region = self.observation_region
        if (region is not None) != required:
            raise ValueError("transport observation region differs from immutable preparation policy")
        if region is not None and (region.generation != generation or region.owner_producer != owner_producer):
            raise ValueError("transport observation region differs from prepared owner or Generation")
