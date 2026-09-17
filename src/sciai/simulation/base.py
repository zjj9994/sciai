"""Simulation adapter contract shared by traditional and learned solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sciai.exceptions import ValidationError


class AdapterState(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class SimulationContext:
    run_id: str = "default"
    working_directory: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


class SimulationAdapter(ABC):
    """Uniform stateful adapter for a simulator or AI surrogate."""

    name = "simulation"
    domain = "general"

    def __init__(self) -> None:
        self.state = AdapterState.CREATED

    def _ensure_active(self) -> None:
        if self.state not in {AdapterState.INITIALIZED, AdapterState.RUNNING}:
            raise ValidationError(
                f"Adapter {self.name!r} cannot read/write in state {self.state.value}; "
                "initialize it first"
            )

    def initialize(self, context: SimulationContext | None = None) -> None:
        if self.state is not AdapterState.CREATED:
            raise ValidationError(f"Cannot initialize adapter in state {self.state.value}")
        self.context = context or SimulationContext()
        self.state = AdapterState.INITIALIZED

    @abstractmethod
    def step(self, step: int, time: float | None = None) -> None:
        """Advance the simulator by one coupling step."""

    @abstractmethod
    def read(self, variables: tuple[str, ...]) -> Mapping[str, Any]:
        """Read interface variables from the simulator."""

    @abstractmethod
    def write(self, values: Mapping[str, Any]) -> None:
        """Write interface variables into the simulator."""

    def snapshot(self) -> Mapping[str, Any]:
        """Return a copy of the adapter's interface state for inspection.

        The base implementation has no interface variables and returns an empty
        mapping. Concrete adapters override this to expose their current state.
        """
        return {}

    def finalize(self) -> None:
        if self.state not in {AdapterState.INITIALIZED, AdapterState.RUNNING}:
            raise ValidationError(f"Cannot finalize adapter in state {self.state.value}")
        self.state = AdapterState.FINISHED


class InMemoryAdapter(SimulationAdapter):
    """Deterministic adapter for tests, prototypes, and learned surrogates."""

    def __init__(
        self,
        name: str,
        domain: str,
        values: Mapping[str, Any] | None = None,
        update: Any | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.domain = domain
        self.values = dict(values or {})
        self.update = update

    def step(self, step: int, time: float | None = None) -> None:
        if self.state not in {AdapterState.INITIALIZED, AdapterState.RUNNING}:
            raise ValidationError("Adapter must be initialized before stepping")
        self.state = AdapterState.RUNNING
        if self.update is not None:
            updated = self.update(dict(self.values), step=step, time=time)
            if updated is not None:
                self.values.update(updated)

    def read(self, variables: tuple[str, ...]) -> Mapping[str, Any]:
        self._ensure_active()
        missing = set(variables) - set(self.values)
        if missing:
            raise ValidationError(f"Adapter {self.name!r} cannot provide {sorted(missing)}")
        return {name: self.values[name] for name in variables}

    def write(self, values: Mapping[str, Any]) -> None:
        self._ensure_active()
        self.values.update(values)

    def snapshot(self) -> Mapping[str, Any]:
        return dict(self.values)
