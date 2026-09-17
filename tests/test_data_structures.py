import numpy as np
import pytest

from sciai import (
    GraphData,
    MeshData,
    ParticleSystem,
    ScalarField,
    Spectrum,
    TensorField,
    TimeSeries,
    VectorField,
)
from sciai.exceptions import ValidationError


def test_eight_scientific_data_types_construct() -> None:
    scalar = ScalarField(values=np.zeros((2, 3)), unit="K")
    vector = VectorField(values=np.zeros((2, 3, 2)), unit="m/s")
    tensor = TensorField(values=np.zeros((2, 3, 2, 2)), unit="Pa")
    graph = GraphData(
        values=np.ones((3, 2)),
        edge_index=np.asarray([[0, 1], [1, 2]]),
    )
    particles = ParticleSystem(values=np.zeros((4, 3)), unit="m")
    mesh = MeshData(
        values=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        unit="m",
        cells={"triangle": [[0, 1, 2]]},
    )
    series = TimeSeries(values=np.asarray([[1.0], [2.0]]), time=[0.0, 1.0])
    spectrum = Spectrum(values=[2.0, 3.0], axis=[10.0, 20.0])

    assert scalar.shape == (2, 3)
    assert vector.component_axis == 2
    assert tensor.tensor_rank == 2
    assert graph.edge_index.shape == (2, 2)
    assert particles.positions.shape == (4, 3)
    assert mesh.cells["triangle"].shape == (1, 3)
    assert series.time_unit.symbol == "s"
    assert spectrum.axis_kind == "frequency"


def test_coordinate_length_is_checked() -> None:
    with pytest.raises(ValidationError):
        ScalarField(values=np.zeros((2, 3)), coordinates=([0.0], [0.0, 1.0, 2.0]))


def test_time_must_increase() -> None:
    with pytest.raises(ValidationError):
        TimeSeries(values=[1.0, 2.0], time=[1.0, 1.0])
