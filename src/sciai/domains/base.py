"""Domain plugin contract and discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from sciai.exceptions import RegistryError
from sciai.registry import (
    Registry,
    data_loader_registry,
    dataset_registry,
    domain_registry,
    model_registry,
    operator_registry,
    pipeline_registry,
    simulation_registry,
    trainer_registry,
)


@dataclass(frozen=True)
class DomainPlugin:
    """A declarative, installable extension for one scientific domain."""

    name: str
    version: str
    description: str
    subdomains: tuple[str, ...] = ()
    data_types: tuple[type[Any], ...] = ()
    models: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, Any] = field(default_factory=dict)
    trainers: dict[str, Any] = field(default_factory=dict)
    operators: dict[str, Callable[..., Any]] = field(default_factory=dict)
    pipelines: dict[str, Any] = field(default_factory=dict)
    simulators: dict[str, Any] = field(default_factory=dict)
    loaders: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _registry_groups(
        self,
    ) -> tuple[tuple[Registry[Any], Mapping[str, Any]], ...]:
        return (
            (model_registry, self.models),
            (dataset_registry, self.datasets),
            (trainer_registry, self.trainers),
            (operator_registry, self.operators),
            (pipeline_registry, self.pipelines),
            (simulation_registry, self.simulators),
            (data_loader_registry, self.loaders),
        )

    def register(self) -> DomainPlugin:
        """Install plugin contributions into process-local registries.

        Registration is conflict-safe: the plugin first validates (preflight) that no
        target key is already bound to a *different* object, then performs the writes.
        Re-registering the exact same plugin object is idempotent; binding a different
        object to an existing key raises :class:`RegistryError` and no entry is written.
        """
        self._preflight_registration()
        self._apply_registration()
        return self

    def _preflight_registration(self) -> None:
        if domain_registry.contains(self.name) and domain_registry.get(self.name) is not self:
            raise RegistryError(
                f"Domain {self.name!r} is already registered (different plugin object)"
            )
        for registry, entries in self._registry_groups():
            for key, value in entries.items():
                if registry.contains(key) and registry.get(key) is not value:
                    raise RegistryError(
                        f"Key {key!r} from plugin {self.name!r} conflicts with an existing "
                        f"entry in {registry.name!r}"
                    )

    def _apply_registration(self) -> None:
        if not domain_registry.contains(self.name):
            domain_registry.register(self.name, self)
        for registry, entries in self._registry_groups():
            for key, value in entries.items():
                if not registry.contains(key):
                    registry.register(key, value)


def discover_plugins(*, group: str = "sciai.domains") -> tuple[DomainPlugin, ...]:
    """Discover and register installed plugins through Python entry points."""
    discovered: list[DomainPlugin] = []
    for entry_point in entry_points(group=group):
        try:
            candidate = entry_point.load()
        except Exception as exc:
            raise RegistryError(
                f"Failed to load plugin entry point {entry_point.name!r}"
            ) from exc
        plugin = (
            candidate()
            if callable(candidate) and not isinstance(candidate, DomainPlugin)
            else candidate
        )
        if not isinstance(plugin, DomainPlugin):
            raise RegistryError(
                f"Entry point {entry_point.name!r} did not provide a DomainPlugin"
            )
        try:
            plugin.register()
        except Exception as exc:
            raise RegistryError(
                f"Failed to register plugin {entry_point.name!r}"
            ) from exc
        discovered.append(plugin)
    return tuple(discovered)
