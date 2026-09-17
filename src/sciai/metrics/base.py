"""Streaming metric protocol and collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class Metric(ABC):
    """Stateful metric suitable for batch-wise scientific evaluation."""

    @abstractmethod
    def update(self, prediction: Any, target: Any) -> None:
        """Accumulate one batch."""

    @abstractmethod
    def compute(self) -> float:
        """Return the aggregate value."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state."""


class MetricCollection:
    """Update and compute a named set of metrics together."""

    def __init__(self, metrics: Mapping[str, Metric]) -> None:
        self.metrics = dict(metrics)

    def update(self, prediction: Any, target: Any) -> None:
        for metric in self.metrics.values():
            metric.update(prediction, target)

    def compute(self) -> dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
