import hashlib
import struct
import zlib
from pathlib import Path
from uuid import UUID

from xtest.harness.sglang.serving.graph import read_graph_events


def test_graph_reader_preserves_registered_template_and_execution(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[6] / "resources/observability/v7/metadata.json"
    metadata = fixture.read_bytes()
    (tmp_path / "metadata.3.1.json").write_bytes(metadata)
    chunks = tmp_path / "chunks/3"
    chunks.mkdir(parents=True)
    payload = bytearray(256)
    struct.pack_into("<QQQII", payload, 0, 19, 0, 2, 504, (1 << 3) | (1 << 4))
    struct.pack_into("<QQQ", payload, 64, 1, 1, 7)
    struct.pack_into("<QQ", payload, 216, 1, 1)
    header = bytearray(128)
    struct.pack_into(
        "<8sHHHHQQQQQQQQQII",
        header,
        0,
        b"XPOOLC07",
        7,
        128,
        256,
        0,
        3,
        1,
        1,
        1,
        19,
        19,
        0,
        0,
        256,
        zlib.crc32(payload),
        0,
    )
    header[96:112] = UUID("11111111-1111-4111-8111-111111111111").bytes
    header[112:128] = hashlib.sha256(metadata).digest()[:16]
    struct.pack_into("<I", header, 92, zlib.crc32(header))
    (chunks / "1.xpool").write_bytes(header + payload)
    [event] = read_graph_events(tmp_path)
    assert event["event"] == "execute_begin"
    assert event["template_sequence"] == 1
    assert event["execution_sequence"] == 7
    assert event["event_sequence"] == 19
    assert event["backend_class"] == "FixtureBackend"
    assert event["time_ns"] == 0
