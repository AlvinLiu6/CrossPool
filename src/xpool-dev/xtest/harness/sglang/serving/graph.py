"""Read registered v7 SGLang Host Graph evidence for serving assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from xtest.harness.support.native.timeline import read_sources


class GraphObservation(TypedDict):
    """Derived test facts retaining original Event and Graph identities."""

    run_id: str
    producer: int
    event_sequence: int
    template_sequence: int | None
    execution_sequence: int | None
    pid: int
    time_ns: int
    forward_phase: Literal["decode", "prefill"]
    backend_class: str | None
    event: str


def read_graph_events(outdir: Path) -> list[GraphObservation]:
    """Resolve Host Graph events through their exact Producer metadata."""

    events: list[GraphObservation] = []
    names = {
        500: "capture_begin",
        501: "capture_end",
        502: "capture_error",
        503: "template_installed",
        504: "execute_begin",
        505: "execute_end",
        506: "execute_error",
    }
    for source in read_sources(outdir):
        if source.metadata.process.runtime_role != "instance":
            continue
        for event in source.events:
            if event.kind not in names:
                continue
            template = next(
                (
                    template
                    for template in source.metadata.graph_templates
                    if template.owner_producer == event.graph_owner
                    and template.template_sequence == event.graph_template
                ),
                None,
            )
            assert event.data1 in (1, 2)
            events.append(
                {
                    "run_id": str(source.metadata.run_id),
                    "producer": source.metadata.producer_id,
                    "event_sequence": event.event_sequence,
                    "template_sequence": event.graph_template if event.presence & (1 << 3) else None,
                    "execution_sequence": event.graph_execution if event.presence & (1 << 4) else None,
                    "pid": source.metadata.process.pid,
                    "time_ns": event.raw_time_ns,
                    "forward_phase": "decode" if event.data1 == 1 else "prefill",
                    "backend_class": template.backend_class if template else None,
                    "event": names[event.kind],
                }
            )
    return events
