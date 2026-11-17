"""Native Graph structure collection with v7 background attachment publication."""

from xpool.native import RuntimeRole

runtime_roles = frozenset({RuntimeRole.FFNAGENT})


def install() -> None:
    """Runtime metadata preparation and Native installation Hooks own collection."""
