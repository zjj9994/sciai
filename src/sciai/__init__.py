"""sciai: a unified, domain-extensible foundation for AI for Science."""

from sciai.auto import AutoDataset, AutoModel, AutoTrainer
from sciai.core import (
    CoordinateSystem,
    Dimension,
    GraphData,
    MeshData,
    ParticleSystem,
    Quantity,
    ReferenceFrame,
    ScalarField,
    ScienceData,
    ScienceMetadata,
    Spectrum,
    TensorField,
    TimeSeries,
    Unit,
    VectorField,
    units,
)
from sciai.data import DatasetInfo, ScienceDataset
from sciai.metrics import Metric, MetricCollection
from sciai.models import ScienceModel
from sciai.pipelines import SciencePipeline, pipeline
from sciai.training import ScienceTrainer, TrainingConfig
from sciai.workflow import ScienceWorkflow

__version__ = "0.1.0a1"

__all__ = [
    "AutoDataset",
    "AutoModel",
    "AutoTrainer",
    "CoordinateSystem",
    "DatasetInfo",
    "Dimension",
    "GraphData",
    "MeshData",
    "Metric",
    "MetricCollection",
    "ParticleSystem",
    "Quantity",
    "ReferenceFrame",
    "ScalarField",
    "ScienceData",
    "ScienceDataset",
    "ScienceMetadata",
    "ScienceModel",
    "SciencePipeline",
    "ScienceTrainer",
    "ScienceWorkflow",
    "Spectrum",
    "TensorField",
    "TimeSeries",
    "TrainingConfig",
    "Unit",
    "VectorField",
    "pipeline",
    "units",
]
