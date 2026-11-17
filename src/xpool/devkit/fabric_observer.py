"""Immutable Fabric event collection configured during runtime preparation."""

from xpool.native import RuntimeRole

runtime_roles = frozenset({RuntimeRole.ATNAGENT, RuntimeRole.FFNAGENT})


def install() -> None:
    """Runtime preparation installs the shared Native Collector/Writer."""
