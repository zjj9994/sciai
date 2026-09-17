"""Typed providers stored in Auto registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sciai.data import ScienceDataset
from sciai.models import ScienceModel


@dataclass(frozen=True)
class ModelProvider:
    factory: Callable[..., ScienceModel]
    domain: str
    tasks: tuple[str, ...]
    description: str = ""
    pretrained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProvider:
    factory: Callable[..., ScienceDataset[Any]]
    domain: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineProvider:
    factory: Callable[..., Any]
    domains: tuple[str, ...] = ()
    description: str = ""
