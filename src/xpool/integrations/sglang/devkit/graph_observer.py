"""Registered v7 Host Graph events for SGLang runner calls."""

from __future__ import annotations

import atexit
import functools
import logging
import time
from collections.abc import Callable
from typing import Literal
from weakref import WeakKeyDictionary

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

import xpool.native.devkit.graph_observer
from xpool.native import RuntimeRole

__all__ = ["close", "install"]
runtime_roles = frozenset({RuntimeRole.INSTANCE})
logger = logging.getLogger(__name__)
installed = False
enqueue_failures = 0
first_failure = ""

type GraphRunner = DecodeCudaGraphRunner | PrefillCudaGraphRunner
type ForwardPhase = Literal["decode", "prefill"]
type GraphOperation = Literal["capture", "execute"]

templates: WeakKeyDictionary[GraphRunner, int] = WeakKeyDictionary()


def install() -> None:
    """Wrap actual methods; runtime preparation supplies registered Sources."""

    global installed
    if installed:
        return
    for runner in (DecodeCudaGraphRunner, PrefillCudaGraphRunner):
        phase: ForwardPhase = "decode" if runner is DecodeCudaGraphRunner else "prefill"
        setattr(runner, "capture", wrap_graph_method(phase, "capture", runner.capture))
        setattr(runner, "execute", wrap_graph_method(phase, "execute", runner.execute))
    atexit.register(close)
    installed = True


def record_event(runner: GraphRunner, phase: ForwardPhase, kind: int, identity: int, template_sequence: int = 0) -> int:
    """Preserve serving behavior and retain bounded pre-enqueue failure counts."""

    try:
        return xpool.native.devkit.graph_observer.record_host(
            kind,
            1 if phase == "decode" else 2,
            type(runner.backend).__name__,
            identity,
            template_sequence,
            time.monotonic_ns(),
        )
    except Exception as error:
        record_failure(error)
        return 0


def record_failure(error: Exception) -> None:
    """Retain one bounded diagnostic; perform no logging or file I/O in Hooks."""

    global enqueue_failures, first_failure
    enqueue_failures = min(enqueue_failures + 1, 2**64 - 1)
    if not first_failure:
        first_failure = f"{type(error).__name__}: {error}"[:512]


def wrap_graph_method[R](
    phase: ForwardPhase, operation: GraphOperation, original: Callable[..., R]
) -> Callable[..., R]:
    """Preserve the original call and reuse one observation identity per call."""

    @functools.wraps(original)
    def wrapped(runner: GraphRunner, *args: object, **kwargs: object) -> R:
        begin, end, error = (500, 501, 502) if operation == "capture" else (504, 505, 506)
        try:
            template = templates.get(runner, 0) if operation == "execute" else 0
        except TypeError as error:
            record_failure(error)
            template = 0
        identity = record_event(runner, phase, begin, 0, template)
        try:
            result = original(runner, *args, **kwargs)
        except BaseException:
            if identity:
                record_event(runner, phase, error, identity, template)
            raise
        if identity:
            record_event(runner, phase, end, identity, template)
            if operation == "capture":
                try:
                    templates[runner] = identity
                except TypeError as error:
                    record_failure(error)
                record_event(runner, phase, 503, identity)
        return result

    return wrapped


def close() -> bool:
    """Runtime retirement joins the shared service after actual producer retirement."""

    if enqueue_failures:
        logger.warning("Graph event submission failures=%s first_failure=%s", enqueue_failures, first_failure)
    return enqueue_failures == 0
