"""Traditional, learned, and multiphysics simulation integration."""

from typing import Any

from sciai.registry import simulation_registry
from sciai.simulation.base import (
    AdapterState,
    InMemoryAdapter,
    SimulationAdapter,
    SimulationContext,
)
from sciai.simulation.coupling import CoupledResult, CoupledSimulator, Coupling


def _in_memory_provider(*, name: str, domain: str = "general", **kwargs: Any) -> InMemoryAdapter:
    return InMemoryAdapter(name=name, domain=domain, **kwargs)


if not simulation_registry.contains("in-memory"):
    simulation_registry.register("in-memory", _in_memory_provider)

__all__ = [
    "AdapterState",
    "CoupledResult",
    "CoupledSimulator",
    "Coupling",
    "InMemoryAdapter",
    "SimulationAdapter",
    "SimulationContext",
]
