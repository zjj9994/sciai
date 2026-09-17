"""Automatic model, dataset, and trainer resolution."""

from sciai.auto.factory import AutoDataset, AutoModel, AutoTrainer
from sciai.auto.providers import DatasetProvider, ModelProvider, PipelineProvider

__all__ = [
    "AutoDataset",
    "AutoModel",
    "AutoTrainer",
    "DatasetProvider",
    "ModelProvider",
    "PipelineProvider",
]
