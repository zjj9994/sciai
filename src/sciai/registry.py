"""Thread-safe registries used by the core and domain plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from threading import RLock
from typing import Any, Generic, TypeVar

from sciai.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A small explicit registry with deterministic conflict handling."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}
        self._lock = RLock()

    @staticmethod
    def normalize(key: str) -> str:
        normalized = key.strip().lower().replace("_", "-")
        if not normalized:
            raise RegistryError("Registry keys cannot be empty")
        return normalized

    def register(self, key: str, value: T, *, replace: bool = False) -> T:
        normalized = self.normalize(key)
        with self._lock:
            if normalized in self._items and not replace:
                raise RegistryError(f"{normalized!r} is already registered in {self.name}")
            self._items[normalized] = value
        return value

    def decorator(self, key: str, *, replace: bool = False) -> Callable[[T], T]:
        def register_value(value: T) -> T:
            return self.register(key, value, replace=replace)

        return register_value

    def get(self, key: str) -> T:
        normalized = self.normalize(key)
        try:
            return self._items[normalized]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(
                f"{key!r} is not registered in {self.name}. Available: {choices}"
            ) from exc

    def contains(self, key: str) -> bool:
        return self.normalize(key) in self._items

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._items))

    def items(self) -> tuple[tuple[str, T], ...]:
        with self._lock:
            return tuple(sorted(self._items.items()))

    def snapshot(self) -> Mapping[str, T]:
        with self._lock:
            return dict(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


model_registry: Registry[Any] = Registry("models")
dataset_registry: Registry[Any] = Registry("datasets")
trainer_registry: Registry[Any] = Registry("trainers")
pipeline_registry: Registry[Any] = Registry("pipelines")
domain_registry: Registry[Any] = Registry("domains")
operator_registry: Registry[Any] = Registry("operators")
simulation_registry: Registry[Any] = Registry("simulation adapters")
data_loader_registry: Registry[Any] = Registry("data loaders")
