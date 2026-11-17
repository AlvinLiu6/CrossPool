from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import safetensors
import torch

import xpool.native
from xpool.devkit.ffn_routing_observer import encode_routing


@dataclass(frozen=True)
class Attachment:
    sequence: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    generation: int = 3
    instance_index: int = 0
    invocation_sequence: int = 11
    lane_index: int = 0
    lease_sequence: int = 7
    layer_ordinal: int = 2
    pe: int = 1
    clock_domain_id: int = 4
    raw_publication_time_ns: int = 0


def test_routing_v7_preserves_tensor_bytes_and_attachment_identity(tmp_path: Path) -> None:
    ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    record = Attachment(19, ids, weights)
    payload = encode_routing(
        "11111111-1111-4111-8111-111111111111",
        5,
        [cast(xpool.native.devkit.ffn_routing_observer.RoutingAttachment, record)],
    )
    path = tmp_path / "routing.safetensors"
    path.write_bytes(payload)
    with safetensors.safe_open(path, framework="pt", device="cpu") as data:
        metadata = data.metadata()
        assert metadata is not None
        assert set(metadata) == {"contract_version", "run_id", "producer_id", "records"}
        assert metadata["contract_version"] == "7" and metadata["producer_id"] == "5"
        records = json.loads(metadata["records"])
        assert records[0]["sequence"] == 19
        assert records[0]["owner"] == 5
        assert records[0]["raw_publication_time_ns"] == 0
        assert records[0]["lane_index"] == 0 and records[0]["lease_sequence"] == 7
        assert records[0]["shape"] == [2, 2]
        assert torch.equal(data.get_tensor("records.0.topk_ids"), ids)
        assert torch.equal(data.get_tensor("records.0.topk_weights"), weights)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "contiguity"])
def test_routing_invalid_tensor_fails_explicitly(invalid: str) -> None:
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    weights = torch.ones((2, 2), dtype=torch.float32)
    if invalid == "dtype":
        ids = ids.to(torch.int64)
    elif invalid == "shape":
        weights = weights[:1]
    else:
        ids = ids.t()
    record = cast(xpool.native.devkit.ffn_routing_observer.RoutingAttachment, Attachment(1, ids, weights))
    with pytest.raises(ValueError, match="contiguous CPU"):
        encode_routing("11111111-1111-4111-8111-111111111111", 5, [record])
