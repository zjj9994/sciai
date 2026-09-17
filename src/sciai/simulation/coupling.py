"""Explicit multiphysics coupling graph and execution engine."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sciai.exceptions import CouplingError, RegistryError, ValidationError
from sciai.registry import simulation_registry
from sciai.simulation.base import SimulationAdapter, SimulationContext

Transfer = Callable[[Any], Any]


@dataclass(frozen=True)
class Coupling:
    source: str
    target: str
    variables: tuple[str, ...]
    interface: str = "default"
    transforms: Mapping[str, Transfer] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValidationError("A coupling source and target must differ")
        if not self.variables:
            raise ValidationError("A coupling must exchange at least one variable")
        unknown = set(self.transforms) - set(self.variables)
        if unknown:
            raise ValidationError(f"Transforms reference unknown variables: {sorted(unknown)}")


@dataclass(frozen=True)
class CoupledResult:
    steps: int
    adapter_states: dict[str, dict[str, Any]]


class CoupledSimulator:
    """Run a directed graph of simulator adapters with typed exchange points."""

    def __init__(
        self,
        adapters: Sequence[
            SimulationAdapter | tuple[str, str] | tuple[str, str, Mapping[str, Any]]
        ],
    ) -> None:
        self.adapters: dict[str, SimulationAdapter] = {}
        for specification in adapters:
            adapter = self._resolve_adapter(specification)
            if adapter.name in self.adapters:
                raise CouplingError(f"Duplicate adapter name {adapter.name!r}")
            self.adapters[adapter.name] = adapter
        if not self.adapters:
            raise CouplingError("CoupledSimulator requires at least one adapter")
        self.couplings: list[Coupling] = []

    @staticmethod
    def _resolve_adapter(
        specification: SimulationAdapter | tuple[str, str] | tuple[str, str, Mapping[str, Any]],
    ) -> SimulationAdapter:
        if isinstance(specification, SimulationAdapter):
            return specification
        if len(specification) not in {2, 3}:
            raise CouplingError("Adapter tuples must be (name, backend[, config])")
        name, backend = specification[:2]
        config = dict(specification[2]) if len(specification) == 3 else {}
        try:
            provider = simulation_registry.get(backend)
        except RegistryError as exc:
            raise CouplingError(
                f"Simulation backend {backend!r} is not installed. Register an adapter "
                "plugin for this external solver before constructing the coupling."
            ) from exc
        adapter = provider(name=name, **config)
        if not isinstance(adapter, SimulationAdapter):
            raise CouplingError(f"Backend {backend!r} did not create a SimulationAdapter")
        return adapter

    def couple(
        self,
        source: str,
        target: str,
        *,
        variables: Sequence[str],
        interface: str = "default",
        transforms: Mapping[str, Transfer] | None = None,
    ) -> CoupledSimulator:
        missing = {source, target} - set(self.adapters)
        if missing:
            raise CouplingError(f"Unknown coupling adapters: {sorted(missing)}")
        self.couplings.append(
            Coupling(
                source=source,
                target=target,
                variables=tuple(variables),
                interface=interface,
                transforms=transforms or {},
            )
        )
        return self

    def set_coupling(
        self,
        *,
        interface: str,
        exchange: Sequence[str],
    ) -> CoupledSimulator:
        """Connect adjacent adapters bidirectionally using the concise public API."""
        names = tuple(self.adapters)
        if len(names) < 2:
            raise CouplingError("set_coupling requires at least two adapters")
        for source, target in zip(names, names[1:], strict=False):
            self.couple(source, target, variables=exchange, interface=interface)
            self.couple(target, source, variables=exchange, interface=interface)
        return self

    def validate(self) -> None:
        for coupling in self.couplings:
            source = self.adapters[coupling.source]
            try:
                source.read(coupling.variables)
            except Exception as exc:
                raise CouplingError(
                    f"Coupling {coupling.source!r} -> {coupling.target!r} cannot read "
                    f"{coupling.variables} on interface {coupling.interface!r}"
                ) from exc

    def run(
        self,
        *,
        steps: int,
        time_step: float | None = None,
        context: SimulationContext | None = None,
        scheme: str = "jacobi",
    ) -> CoupledResult:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise ValidationError(
                f"Coupling steps must be an int, got {type(steps).__name__}"
            )
        if steps < 1:
            raise ValidationError("Coupling steps must be positive")
        if time_step is not None:
            if isinstance(time_step, bool) or not isinstance(time_step, (int, float)):
                raise ValidationError(
                    f"Coupling time_step must be a number, got {type(time_step).__name__}"
                )
            if not math.isfinite(time_step) or time_step <= 0:
                raise ValidationError("Coupling time_step must be finite and positive")
        if scheme not in {"jacobi", "gauss-seidel"}:
            raise ValidationError(f"Unknown coupling scheme {scheme!r}")

        initialized: list[SimulationAdapter] = []
        try:
            for adapter in self.adapters.values():
                adapter.initialize(context)
                initialized.append(adapter)
        except Exception:
            for adapter in initialized:
                with contextlib.suppress(Exception):
                    adapter.finalize()
            raise

        try:
            self.validate()
            for step in range(steps):
                current_time = None if time_step is None else step * time_step
                for adapter in self.adapters.values():
                    adapter.step(step, current_time)
                if scheme == "jacobi":
                    self._exchange_jacobi()
                else:
                    self._exchange_gauss_seidel()
            return CoupledResult(
                steps=steps,
                adapter_states={
                    name: dict(adapter.snapshot())
                    for name, adapter in self.adapters.items()
                },
            )
        finally:
            for adapter in self.adapters.values():
                with contextlib.suppress(Exception):
                    adapter.finalize()

    def _exchange_jacobi(self) -> None:
        payloads: list[tuple[Coupling, Mapping[str, Any]]] = []
        for coupling in self.couplings:
            source = self.adapters[coupling.source]
            try:
                payload = dict(source.read(coupling.variables))
            except Exception as exc:
                raise CouplingError(
                    f"Coupling {coupling.source!r} -> {coupling.target!r} failed to read "
                    f"{coupling.variables} on interface {coupling.interface!r}"
                ) from exc
            for variable, transform in coupling.transforms.items():
                payload[variable] = transform(payload[variable])
            payloads.append((coupling, payload))
        for coupling, outbound in payloads:
            target = self.adapters[coupling.target]
            try:
                target.write(dict(outbound))
            except Exception as exc:
                raise CouplingError(
                    f"Coupling {coupling.source!r} -> {coupling.target!r} failed to write "
                    f"{tuple(outbound)} on interface {coupling.interface!r}"
                ) from exc

    def _exchange_gauss_seidel(self) -> None:
        for coupling in self.couplings:
            source = self.adapters[coupling.source]
            try:
                payload = dict(source.read(coupling.variables))
            except Exception as exc:
                raise CouplingError(
                    f"Coupling {coupling.source!r} -> {coupling.target!r} failed to read "
                    f"{coupling.variables} on interface {coupling.interface!r}"
                ) from exc
            for variable, transform in coupling.transforms.items():
                payload[variable] = transform(payload[variable])
            target = self.adapters[coupling.target]
            try:
                target.write(dict(payload))
            except Exception as exc:
                raise CouplingError(
                    f"Coupling {coupling.source!r} -> {coupling.target!r} failed to write "
                    f"{tuple(payload)} on interface {coupling.interface!r}"
                ) from exc
