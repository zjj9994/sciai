"""Contract tests for the hardened scientific core (ontology, data, operators, io)."""

import numpy as np
import pytest

from sciai import (
    CoordinateSystem,
    Dimension,
    GraphData,
    MeshData,
    ParticleSystem,
    Quantity,
    ReferenceFrame,
    ScalarField,
    Spectrum,
    TensorField,
    VectorField,
    units,
)
from sciai.exceptions import ValidationError
from sciai.io import load_data
from sciai.operators import (
    curl,
    divergence,
    gradient,
    integrate,
    laplacian,
)


# --------------------------------------------------------------------------- #
# Unit registry parsing hardening
# --------------------------------------------------------------------------- #
def test_parse_rejects_malformed_operators() -> None:
    for bad in ["", "*m", "m*", "m//s", "/", "m^", "m^^2", "m*/s"]:
        with pytest.raises(ValidationError):
            units.parse(bad)


def test_parse_keeps_valid_aliases() -> None:
    assert units.parse("m/s^2").dimension == Dimension({"length": 1, "time": -2})
    assert units.parse("m**2").dimension == Dimension({"length": 2})
    assert units.parse("m/s").dimension == Dimension({"length": 1, "time": -1})


def test_derived_units_exposed() -> None:
    assert units.parse("velocity").dimension == Dimension({"length": 1, "time": -1})
    assert units.parse("acceleration").dimension == Dimension({"length": 1, "time": -2})


# --------------------------------------------------------------------------- #
# Reference frame hardening
# --------------------------------------------------------------------------- #
def test_reference_frame_valid_default() -> None:
    frame = ReferenceFrame()
    assert np.allclose(np.asarray(frame.orientation), np.eye(3))


def test_reference_frame_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(origin=(0.0, 0.0, float("inf")))


def test_reference_frame_rejects_non_orthogonal() -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(orientation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)))


def test_reference_frame_rejects_non_square() -> None:
    with pytest.raises(ValidationError):
        ReferenceFrame(orientation=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))


# --------------------------------------------------------------------------- #
# Coordinate system hardening
# --------------------------------------------------------------------------- #
def test_coordinate_system_axes_nonempty() -> None:
    with pytest.raises(ValidationError):
        CoordinateSystem(axes=())


def test_coordinate_system_length_based_flag() -> None:
    cs = CoordinateSystem(axes=("x", "y"), axis_units=("m", "m"))
    assert cs.length_based
    assert cs.is_consistent
    mixed = CoordinateSystem(axes=("x", "t"), axis_units=("m", "s"))
    assert not mixed.length_based
    assert not mixed.is_consistent


# --------------------------------------------------------------------------- #
# Quantity arithmetic hardening
# --------------------------------------------------------------------------- #
def test_quantity_add_sub_returns_not_implemented_for_non_quantity() -> None:
    q = Quantity([1.0], "m")
    assert q.__add__("x") is NotImplemented
    assert q.__sub__(2.0) is NotImplemented
    with pytest.raises(TypeError):
        _ = q + 2.0


def test_quantity_rejects_string_operands() -> None:
    q = Quantity([1.0], "m")
    with pytest.raises(ValidationError):
        _ = q * "2"
    with pytest.raises(ValidationError):
        _ = q / "2"


def test_quantity_allows_complex_values() -> None:
    q = Quantity([1.0 + 2.0j], "m")
    assert np.iscomplexobj(q.value)


def test_quantity_multiplies_by_scalar() -> None:
    q = Quantity([1.0, 2.0], "m")
    assert np.allclose((q * 3).value, [3.0, 6.0])
    assert np.allclose((3 * q).value, [3.0, 6.0])


# --------------------------------------------------------------------------- #
# Data class hardening
# --------------------------------------------------------------------------- #
def test_graphdata_allows_1d_node_features() -> None:
    graph = GraphData(values=np.ones(3), edge_index=np.asarray([[0, 1], [1, 2]]))
    assert graph.values.ndim == 1


def test_graphdata_rejects_3d_node_features() -> None:
    with pytest.raises(ValidationError):
        GraphData(values=np.ones((3, 2, 2)), edge_index=np.asarray([[0, 1], [1, 2]]))


def test_graphdata_edge_features_shape() -> None:
    graph = GraphData(
        values=np.ones((3, 2)),
        edge_index=np.asarray([[0, 1], [1, 2]]),
        edge_features=np.ones((2, 4)),
    )
    assert graph.edge_features.shape == (2, 4)
    with pytest.raises(ValidationError):
        GraphData(
            values=np.ones((3, 2)),
            edge_index=np.asarray([[0, 1], [1, 2]]),
            edge_features=np.ones(2),
        )
    with pytest.raises(ValidationError):
        GraphData(
            values=np.ones((3, 2)),
            edge_index=np.asarray([[0, 1], [1, 2]]),
            edge_features=np.ones((3, 4)),
        )


def test_particle_system_rejects_inf_but_allows_nan() -> None:
    with pytest.raises(ValidationError):
        ParticleSystem(values=np.array([[1.0, float("inf"), 0.0]]))
    # NaN is a legitimate missing-value marker and must be allowed.
    nan_field = ParticleSystem(values=np.array([[1.0, float("nan"), 0.0]]))
    assert nan_field.positions.shape == (1, 3)


def test_mesh_rejects_inf_but_allows_nan() -> None:
    with pytest.raises(ValidationError):
        MeshData(values=np.array([[float("inf"), 0.0]]), cells={"tri": [[0, 0]]})
    MeshData(values=np.array([[float("nan"), 0.0]]), cells={"tri": [[0, 0]]})


def test_spectrum_axis_must_be_monotonic() -> None:
    with pytest.raises(ValidationError):
        Spectrum(values=[2.0, 3.0, 4.0], axis=[10.0, 20.0, 15.0])


def test_vectorfield_component_axis_first_binds_coordinates() -> None:
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 4)
    field = VectorField(values=np.zeros((3, 5, 4)), component_axis=0, coordinates=(x, y))
    assert field.component_axis == 0
    assert field.coordinates is not None
    assert field.coordinates[0].shape == (5,)
    assert field.coordinates[1].shape == (4,)


def test_tensorfield_binds_spatial_coordinates_only() -> None:
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 4)
    field = TensorField(values=np.zeros((5, 4, 2, 2)), tensor_rank=2, coordinates=(x, y))
    assert field.coordinates is not None
    assert field.coordinates[0].shape == (5,)
    assert field.coordinates[1].shape == (4,)


def test_coordinate_system_count_consistency() -> None:
    cs = CoordinateSystem(axes=("x", "y"), axis_units=("m", "m"))
    with pytest.raises(ValidationError):
        ScalarField(
            values=np.zeros((4, 5)),
            coordinates=(np.linspace(0, 1, 4),),
            coordinate_system=cs,
        )


# --------------------------------------------------------------------------- #
# Operator unit propagation
# --------------------------------------------------------------------------- #
def test_gradient_unit_propagation() -> None:
    cs = CoordinateSystem(axes=("x", "y"), axis_units=("m", "m"))
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = ScalarField(values=xx**2 + yy**2, unit="K", coordinates=(x, y), coordinate_system=cs)
    grad = gradient(field, edge_order=2)
    assert grad.unit.symbol == "K/m"
    np.testing.assert_allclose(grad.values[5:-5, 5:-5, 0], 2 * xx[5:-5, 5:-5], atol=1e-8)


def test_laplacian_unit_propagation() -> None:
    cs = CoordinateSystem(axes=("x", "y"), axis_units=("m", "m"))
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = ScalarField(values=xx**2 + yy**2, unit="K", coordinates=(x, y), coordinate_system=cs)
    lap = laplacian(field, edge_order=2)
    assert lap.unit.symbol == "K/m^2"
    np.testing.assert_allclose(lap.values[5:-5, 5:-5], 4.0, atol=1e-8)


def test_divergence_curl_unit_propagation() -> None:
    cs = CoordinateSystem(axes=("x", "y"), axis_units=("m", "m"))
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = VectorField(
        values=np.stack([xx, -yy], axis=-1),
        unit="m/s",
        coordinates=(x, y),
        coordinate_system=cs,
    )
    div = divergence(field, edge_order=2)
    assert div.unit.dimension == Dimension({"time": -1})
    np.testing.assert_allclose(div.values, 0.0, atol=1e-8)

    cs3 = CoordinateSystem(axes=("x", "y", "z"), axis_units=("m", "m", "m"))
    axis_ = np.linspace(-1.0, 1.0, 7)
    x3, y3, z3 = np.meshgrid(axis_, axis_, axis_, indexing="ij")
    rot = VectorField(
        values=np.stack([-y3, x3, np.zeros_like(z3)], axis=-1),
        unit="m/s",
        coordinates=(axis_, axis_, axis_),
        coordinate_system=cs3,
    )
    out = curl(rot, edge_order=2)
    assert out.unit.dimension == Dimension({"time": -1})


def test_unit_propagation_requires_identical_axis_units() -> None:
    cs = CoordinateSystem(axes=("x", "t"), axis_units=("m", "s"))
    x = np.linspace(0.0, 1.0, 11)
    t = np.linspace(0.0, 1.0, 11)
    field = ScalarField(values=xx_dummy(x, t), unit="K", coordinates=(x, t), coordinate_system=cs)
    with pytest.raises(ValidationError):
        gradient(field)


def test_no_coordinate_system_is_dimensionless() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = ScalarField(values=xx**2 + yy**2, coordinates=(x, y))
    grad = gradient(field, edge_order=2)
    assert grad.unit.symbol == "1"


# --------------------------------------------------------------------------- #
# Operator validity checks: nonuniform / bad edge order
# --------------------------------------------------------------------------- #
def test_gradient_rejects_non_monotonic_coordinate() -> None:
    good = np.linspace(0.0, 1.0, 4)
    bad = np.array([0.0, 2.0, 1.0, 3.0])
    field = ScalarField(values=np.zeros((4, 4)), coordinates=(good, bad))
    with pytest.raises(ValidationError):
        gradient(field)


def test_edge_order_must_be_valid() -> None:
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    field = ScalarField(values=np.zeros((11, 11)), coordinates=(x, y))
    with pytest.raises(ValidationError):
        gradient(field, edge_order=3)
    with pytest.raises(ValidationError):
        laplacian(field, edge_order=0)


def test_edge_order_two_requires_three_points() -> None:
    coord = np.array([0.0, 1.0])
    field = ScalarField(values=np.zeros((2, 2)), coordinates=(coord, coord))
    with pytest.raises(ValidationError):
        gradient(field, edge_order=2)


# --------------------------------------------------------------------------- #
# Divergence with component axis not last
# --------------------------------------------------------------------------- #
def test_divergence_component_axis_first() -> None:
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 6)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.stack([xx, -yy], axis=0)  # shape (2, 6, 6), component along axis 0
    field = VectorField(values=values, component_axis=0, coordinates=(x, y))
    div = divergence(field, edge_order=2)
    np.testing.assert_allclose(div.values, 0.0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Integrate returns Quantity for ScienceData, array for ndarray
# --------------------------------------------------------------------------- #
def test_integrate_returns_quantity_with_units() -> None:
    cs = CoordinateSystem(axes=("t",), axis_units=("s",))
    field = ScalarField(
        values=np.array([1.0, 2.0, 3.0]),
        unit="m",
        coordinates=(np.array([0.0, 1.0, 2.0]),),
        coordinate_system=cs,
    )
    result = integrate(field)
    assert isinstance(result, Quantity)
    assert result.unit.symbol == "m*s"
    np.testing.assert_allclose(result.value, 4.0)


def test_integrate_plain_ndarray_returns_array() -> None:
    out = integrate(np.array([0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, 2.0)


# --------------------------------------------------------------------------- #
# Loaders hardening
# --------------------------------------------------------------------------- #
def test_csv_empty_rejected(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValidationError):
        load_data(path)


def test_csv_header_only_rejected(tmp_path) -> None:
    path = tmp_path / "header.csv"
    path.write_text("a,b,c\n")
    with pytest.raises(ValidationError):
        load_data(path)


def test_csv_non_numeric_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,x\n")
    with pytest.raises(ValidationError):
        load_data(path)


def test_csv_ok(tmp_path) -> None:
    path = tmp_path / "ok.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    field = load_data(path)
    assert field.shape == (2, 2)


def test_format_key_is_lowercased(tmp_path) -> None:
    path = tmp_path / "UPPER.CSV"
    path.write_text("a,b\n1,2\n3,4\n")
    field = load_data(path)
    assert field.shape == (2, 2)


def test_npy_object_dtype_rejected(tmp_path) -> None:
    path = tmp_path / "obj.npy"
    np.save(path, np.array(["a", "b"], dtype=object))
    with pytest.raises(ValidationError):
        load_data(path)


def test_unknown_loader_kwargs_rejected(tmp_path) -> None:
    path = tmp_path / "f.npy"
    np.save(path, np.arange(6).reshape(2, 3))
    with pytest.raises(ValidationError):
        load_data(path, bogus=1)


def xx_dummy(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    xx, _ = np.meshgrid(x, t, indexing="ij")
    return xx
