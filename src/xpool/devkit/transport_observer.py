"""Immutable Transport event collection configured during runtime preparation."""

from xpool.native import RuntimeRole

runtime_roles = frozenset({RuntimeRole.INSTANCE, RuntimeRole.ATNAGENT})


def install() -> None:
    """Runtime preparation installs the shared Native Collector/Writer."""
