from __future__ import annotations

from typing import cast

import pytest

import xpool.integrations.sglang.devkit.graph_observer
from xtest.harness.support.sglang.graph_observer import FakeDecodeCudaGraphRunnerState, reset_graph_observer

pytestmark = pytest.mark.usefixtures(reset_graph_observer.__name__)


def test_graph_call_uses_fresh_execution_and_retains_template(monkeypatch: pytest.MonkeyPatch) -> None:
    observer = xpool.integrations.sglang.devkit.graph_observer
    emitted: list[tuple[int, int, int]] = []
    next_identity = iter((10, 20, 21))

    def record(kind: int, phase: int, backend: str, identity: int, template: int, time_ns: int) -> int:
        assigned = identity or next(next_identity)
        emitted.append((kind, assigned, template))
        return assigned

    monkeypatch.setattr(observer.xpool.native.devkit.graph_observer, "record_host", record)
    runner = cast(observer.GraphRunner, FakeDecodeCudaGraphRunnerState())
    capture = observer.wrap_graph_method("decode", "capture", lambda runner: "captured")
    execute = observer.wrap_graph_method("decode", "execute", lambda runner: "result")
    assert capture(runner) == "captured"
    assert execute(runner) == execute(runner) == "result"
    assert emitted == [
        (500, 10, 0),
        (501, 10, 0),
        (503, 10, 0),
        (504, 20, 10),
        (505, 20, 10),
        (504, 21, 10),
        (505, 21, 10),
    ]


def test_graph_failure_preserves_original_error(monkeypatch: pytest.MonkeyPatch) -> None:
    observer = xpool.integrations.sglang.devkit.graph_observer
    emitted: list[int] = []

    def record(kind: int, phase: int, backend: str, identity: int, template: int, time_ns: int) -> int:
        emitted.append(kind)
        return identity or 7

    monkeypatch.setattr(observer.xpool.native.devkit.graph_observer, "record_host", record)
    error = ValueError("runner failed")

    def fail(runner: observer.GraphRunner) -> None:
        raise error

    runner = cast(observer.GraphRunner, FakeDecodeCudaGraphRunnerState())
    with pytest.raises(ValueError) as caught:
        observer.wrap_graph_method("prefill", "capture", fail)(runner)
    assert caught.value is error
    assert emitted == [500, 502]


def test_enqueue_failure_keeps_serving_result_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    observer = xpool.integrations.sglang.devkit.graph_observer

    def fail(*args: object) -> int:
        raise RuntimeError("Source unavailable")

    monkeypatch.setattr(observer.xpool.native.devkit.graph_observer, "record_host", fail)
    runner = cast(observer.GraphRunner, FakeDecodeCudaGraphRunnerState())
    assert observer.wrap_graph_method("decode", "execute", lambda runner: "ok")(runner) == "ok"
    assert observer.enqueue_failures == 1
    assert not observer.close()
    assert "Source unavailable" in caplog.text
