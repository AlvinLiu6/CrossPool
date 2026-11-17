"""Install isolated graph-observer fixtures and inspect their event streams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import pytest

import xpool.integrations.sglang.devkit.graph_observer


@pytest.fixture
def reset_graph_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    observer = xpool.integrations.sglang.devkit.graph_observer
    monkeypatch.setattr(observer, "installed", False)
    monkeypatch.setattr(observer, "enqueue_failures", 0)
    monkeypatch.setattr(observer, "first_failure", "")
    observer.templates.clear()
    yield
    observer.templates.clear()


class FullCudaGraphBackend:
    """Fake backend whose name matches SGLang's full backend evidence."""


class FakeDecodeCudaGraphRunnerState:
    def __init__(self) -> None:
        self.backend = FullCudaGraphBackend()


class BreakableCudaGraphBackend:
    """Fake backend whose name matches SGLang's Breakable backend evidence."""


class FakePrefillCudaGraphRunnerState:
    def __init__(self) -> None:
        self.backend = BreakableCudaGraphBackend()


class SuccessfulGraphRunner(Protocol):
    """Call surface exercised after observer installation."""

    def capture(self) -> str: ...

    def execute(self) -> str: ...


def successful_runner_classes() -> tuple[type[SuccessfulGraphRunner], type[SuccessfulGraphRunner]]:
    """Return decode and prefill runner classes with deterministic calls."""

    class DecodeCudaGraphRunner(FakeDecodeCudaGraphRunnerState):
        def capture(self) -> str:
            return "captured"

        def execute(self) -> str:
            return "executed"

    class PrefillCudaGraphRunner(FakePrefillCudaGraphRunnerState):
        def capture(self) -> str:
            return "prefill-captured"

        def execute(self) -> str:
            return "prefill-executed"

    return DecodeCudaGraphRunner, PrefillCudaGraphRunner


def install_fake_sglang_runner_classes(
    monkeypatch: pytest.MonkeyPatch,
    decode_cuda_graph_runner: type[object],
    prefill_cuda_graph_runner: type[object],
) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "DecodeCudaGraphRunner",
        decode_cuda_graph_runner,
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.devkit.graph_observer,
        "PrefillCudaGraphRunner",
        prefill_cuda_graph_runner,
    )
