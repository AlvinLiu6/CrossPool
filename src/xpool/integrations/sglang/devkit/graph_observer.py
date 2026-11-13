"""SGLang Devkit observer for CUDA Graph capture and execution events."""

from __future__ import annotations

import atexit
import functools
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Literal

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

import xpool.native.devkit.graph_observer
from xpool.cext import ensure_native_loaded
from xpool.config import get_global_config
from xpool.native import RuntimeRole

__all__ = ["close", "install"]

runtime_roles = frozenset({RuntimeRole.INSTANCE})

logger = logging.getLogger(__name__)

type GraphRunner = DecodeCudaGraphRunner | PrefillCudaGraphRunner
type ForwardPhase = Literal["decode", "prefill"]
type GraphOperation = Literal["capture", "execute"]

event_file: Path | None = None
log_session: xpool.native.devkit.graph_observer.LogSession | None = None
install_lock = Lock()
installed = False
enqueue_failures = 0

_EVENT_KINDS = {
    "capture_begin": 1,
    "capture_end": 2,
    "capture_error": 3,
    "execute_begin": 4,
    "execute_end": 5,
    "execute_error": 6,
}


def install() -> None:
    """Prepare bounded Native background logging and wrap both SGLang runners.

    Config and output path are fixed before Graph capture. Native preparation
    admits all storage and starts one process Collector/Writer pair. File I/O
    happens on the Writer; its failures are available from the session report
    and a close-time warning. Existing files remain untouched.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
        RuntimeError: If configuration, ABI, resource admission or worker startup
            fails, or a repeated installation changes the output path/lifetime.
    """

    config = get_global_config()
    outdir = config.debug.graph_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool graph observer requires debug.graph_observer.outdir")

    if config.debug.timeline.enable:
        raise RuntimeError("xpool unified timeline providers are not prepared")

    global event_file, log_session, installed
    with install_lock:
        desired_event_file = outdir / f"xpool.graph-observer.{os.getpid()}.jsonl"
        if installed:
            if event_file != desired_event_file or log_session is None or log_session.report()["closed"]:
                raise RuntimeError("xpool graph observer requires its original active log session")
            return
        ensure_native_loaded()
        policy = config.debug.timeline
        log_session = xpool.native.devkit.graph_observer.prepare_log(
            str(desired_event_file),
            policy.host_bytes,
            policy.disk_bytes,
            policy.metadata_reserve_bytes,
            policy.collection_interval_ms,
            policy.shutdown_wait_ms,
        )
        event_file = desired_event_file
        atexit.register(close)

        setattr(
            DecodeCudaGraphRunner,
            "capture",
            wrap_graph_method("decode", "capture", DecodeCudaGraphRunner.capture),
        )
        setattr(
            DecodeCudaGraphRunner,
            "execute",
            wrap_graph_method("decode", "execute", DecodeCudaGraphRunner.execute),
        )
        setattr(
            PrefillCudaGraphRunner,
            "capture",
            wrap_graph_method("prefill", "capture", PrefillCudaGraphRunner.capture),
        )
        setattr(
            PrefillCudaGraphRunner,
            "execute",
            wrap_graph_method("prefill", "execute", PrefillCudaGraphRunner.execute),
        )
        installed = True


def wrap_graph_method[R](
    forward_phase: ForwardPhase,
    operation: GraphOperation,
    original: Callable[..., R],
) -> Callable[..., R]:
    """Wrap one SGLang graph method with begin, error, and end events.

    Args:
        forward_phase: Decode or prefill runner being observed.
        operation: Capture or execution operation being observed.
        original: Original unbound method implementation.

    Returns:
        Wrapped method preserving the original callable metadata.
    """

    @functools.wraps(original)
    def wrapped(runner: GraphRunner, *args: object, **kwargs: object) -> R:
        write_event(forward_phase, f"{operation}_begin", runner)
        try:
            result = original(runner, *args, **kwargs)
        except BaseException:
            write_event(forward_phase, f"{operation}_error", runner)
            raise
        write_event(forward_phase, f"{operation}_end", runner)
        return result

    return wrapped


def write_event(forward_phase: ForwardPhase, event: str, runner: GraphRunner) -> None:
    """Try one bounded Native enqueue, preserving runner behavior on failure.

    The Native call retains the GIL, copies a bounded backend label and returns
    promptly on full or closed admission. JSON formatting and I/O run later on
    the process Writer. Drops are counted in the compatibility report.
    """

    global enqueue_failures
    try:
        current = log_session
        if current is None:
            return
        current.record(
            1 if forward_phase == "decode" else 2,
            _EVENT_KINDS[event],
            type(runner.backend).__name__,
            time.monotonic_ns(),
        )
    except Exception:
        # Failure accounting also stays off the I/O path. Native queue drops
        # have their own counters; this covers failures before a Native attempt.
        if enqueue_failures < 2**64 - 1:
            enqueue_failures += 1


def close() -> bool:
    """Stop Graph logging and wait up to the configured observation deadline.

    Returns:
        Whether both Native workers retired. A timeout retains their resources
        and permits a later completion check without renewing the wait. A shared
        service remains live until its raw producers also retire; this call
        stops only Graph admission. Logged counters distinguish dropped or
        failed output from successful worker retirement. Called at process exit;
        callers may invoke it earlier after runner work has stopped.
    """

    current = log_session
    if current is None:
        return True
    try:
        joined = current.close()
        report = current.report()
        if (
            not joined
            or report["error"] is not None
            or report["lost"]
            or report["dropped_full"]
            or report["dropped_contention"]
            or report["dropped_input"]
            or enqueue_failures
        ):
            logger.warning(
                "xpool graph observer closed pid=%s joined=%s enqueue_failures=%s report=%s",
                os.getpid(),
                joined,
                enqueue_failures,
                report,
            )
        return joined
    except Exception:
        logger.warning("failed to close xpool graph observer pid=%s", os.getpid(), exc_info=True)
        return False
