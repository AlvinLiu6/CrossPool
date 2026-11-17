from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtest.harness.support.native.observer import assert_transport_observations
from xtest.harness.support.native.timeline import read_sources

FIXTURES = Path(__file__).resolve().parents[6] / "resources" / "observability" / "v7"


def write_vector(tmp_path: Path) -> Path:
    """Install independent frozen C0 bytes as a one-chunk Run."""

    encoding = json.loads((FIXTURES / "encoding.json").read_bytes())
    vector = next(vector for vector in encoding["vectors"] if vector["id"] == "valid_one_event")
    (tmp_path / "metadata.3.1.json").write_bytes((FIXTURES / "metadata.json").read_bytes())
    path = tmp_path / "chunks" / "3" / "1.xpool"
    path.parent.mkdir(parents=True)
    path.write_bytes(bytes.fromhex(vector["hex"]))
    return path


def test_reads_independent_v7_bytes_without_fabricating_terminal_boundary(tmp_path: Path) -> None:
    write_vector(tmp_path)
    (source,) = read_sources(tmp_path)
    (event,) = source.events
    assert event.kind == 100 and event.operation_sequence == 1
    assert event.raw_time_ns == 100 and event.instance_rank == 0
    with pytest.raises(AssertionError):
        assert_transport_observations(tmp_path, expected_count=1, site="instance")


def test_rejects_corrupted_payload(tmp_path: Path) -> None:
    path = write_vector(tmp_path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(AssertionError):
        read_sources(tmp_path)


def test_rejects_metadata_hash_mismatch(tmp_path: Path) -> None:
    write_vector(tmp_path)
    metadata = tmp_path / "metadata.3.1.json"
    metadata.write_bytes(metadata.read_bytes() + b"\n")
    with pytest.raises(AssertionError):
        read_sources(tmp_path)
