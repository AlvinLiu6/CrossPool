from __future__ import annotations

import pytest
from pydantic import ValidationError

from xpool.transport import TransportArenaHandle, TransportObservationRegion


def test_transport_arena_handle_validates_canonical_hex() -> None:
    handle = TransportArenaHandle(handle="ab" * 64)

    assert handle.handle == "ab" * 64
    with pytest.raises(ValueError, match="128 lowercase hex"):
        TransportArenaHandle(handle="ab")
    with pytest.raises(ValueError, match="only lowercase hex"):
        TransportArenaHandle(handle="AB" * 64)


@pytest.mark.parametrize("version", [6, 7, 8])
def test_observation_ipc_accepts_only_current_contract(version: int) -> None:
    descriptor = {
        "version": version,
        "owner_producer": 2,
        "endpoint_sequence": 1,
        "generation": 1,
        "allocation_bytes": 256,
        "handle": "ab" * 64,
    }
    if version != 7:
        with pytest.raises(ValidationError) as caught:
            TransportObservationRegion.model_validate(descriptor)
        assert caught.value.errors()[0]["loc"] == ("version",)
    else:
        region = TransportObservationRegion.model_validate(descriptor)
        assert region.native().version == 7
