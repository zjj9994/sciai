"""NumPy reference implementations of common mathematical physics operators.

Unit propagation
----------------
When a field carries a :class:`CoordinateSystem`, operators derive units from the
axis units:

* ``gradient``      -> ``U / L``   (per coordinate axis)
* ``divergence``    -> ``U / L``
* ``curl``          -> ``U / L``
* ``laplacian``     -> ``U / L^2``

For a single propagated unit every spatial axis must share an identical unit; otherwise
a :class:`ValidationError` is raised (a field cannot carry a single unit while its axes
differ). When no ``CoordinateSystem`` is present the coordinates are treated as
dimensionless, preserving the previous behaviour and keeping the public API stable.

Fourier / wavelet transforms (``fft``, ``ifft``, ``haar_wavelet``) are returned as raw
arrays. They are *not* unit-preserving physical transforms, so they intentionally carry no
unit and the caller is responsible for their interpretation.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from sciai.core import ScalarField, ScienceData, VectorField
from sciai.core.ontology import Quantity, Unit, units
from sciai.exceptions import ValidationError
from sciai.registry import operator_registry


def _spacing(field: ScalarField | VectorField, axis: int) -> np.ndarray | float:
    """Coordinate array (or 1.0) for value-dimension ``axis`` mapped via spatial axes."""
    if field.coordinates is None:
        return 1.0
    spatial = field._spatial_axis_indices()
    if axis not in spatial:
        return 1.0
    coord_index = spatial.index(axis)
    if coord_index >= len(field.coordinates):
        return 1.0
    return np.asarray(field.coordinates[coord_index])


def _axis_unit(field: ScalarField | VectorField, coord_index: int) -> Unit:
    """Unit of the coordinate at ``coord_index`` or dimensionless when absent."""
    coordinate_system = field.coordinate_system
    if coordinate_system is None or coord_index >= len(coordinate_system.axis_units):
        return units.parse("1")
    return cast(Unit, coordinate_system.axis_units[coord_index])


def _single_spatial_unit(
    field: ScalarField | VectorField, spatial_indices: list[int]
) -> Unit | None:
    """Return the (single) spatial axis unit, or ``None`` when dimensionless.

    Raises when the spatial axis units differ, because a single propagated unit cannot
    be formed from mixed axes.
    """
    coordinate_system = field.coordinate_system
    if coordinate_system is None:
        return None
    axis_units = [cast(Unit, coordinate_system.axis_units[i]) for i in spatial_indices]
    if len({unit.symbol for unit in axis_units}) != 1:
        raise ValidationError(
            "Spatial axis units must be identical for unit propagation; found "
            f"{[unit.symbol for unit in axis_units]}"
        )
    return axis_units[0]


def _validate_differentiation_axes(
    field: ScalarField | VectorField, axes: list[int], edge_order: int
) -> None:
    """Check edge_order and that every used coordinate axis is well-formed."""
    if edge_order not in (1, 2):
        raise ValidationError("edge_order must be 1 or 2 for finite differences")
    if field.coordinates is None:
        return
    spatial = field._spatial_axis_indices()
    for axis in axes:
        if axis not in spatial:
            continue
        coord_index = spatial.index(axis)
        if coord_index >= len(field.coordinates):
            continue
        coord = np.asarray(field.coordinates[coord_index])
        if coord.ndim != 1 or coord.size < 2:
            raise ValidationError("A coordinate axis needs at least two points")
        if edge_order == 2 and coord.size < 3:
            raise ValidationError("edge_order=2 requires at least three coordinate points")
        diff = np.diff(coord)
        if not (np.all(diff > 0) or np.all(diff < 0)):
            raise ValidationError("Coordinate axes must be strictly monotonic")


@operator_registry.decorator("gradient")
def gradient(field: ScalarField, *, edge_order: int = 1) -> VectorField:
    """Compute a finite-difference gradient along every spatial axis."""
    if not isinstance(field, ScalarField):
        raise ValidationError("gradient expects a ScalarField")
    spatial = field._spatial_axis_indices()
    _validate_differentiation_axes(field, spatial, edge_order)
    axes = tuple(spatial)
    spacings = tuple(_spacing(field, axis) for axis in axes)
    derivatives = np.gradient(  # type: ignore[call-overload]
        field.values,
        *spacings,
        axis=axes,
        edge_order=edge_order,
    )
    if isinstance(derivatives, np.ndarray):
        derivatives = [derivatives]
    unit = _single_spatial_unit(field, spatial)
    field_unit = cast(Unit, field.unit)
    result_unit = field_unit / unit if unit is not None else field_unit
    return VectorField(
        values=np.stack(derivatives, axis=-1),
        unit=result_unit,
        coordinates=field.coordinates,
        coordinate_system=field.coordinate_system,
        metadata=field.metadata,
        component_axis=-1,
    )


@operator_registry.decorator("divergence")
def divergence(field: VectorField, *, edge_order: int = 1) -> ScalarField:
    """Compute divergence for a vector field whose components match spatial rank."""
    if not isinstance(field, VectorField):
        raise ValidationError("divergence expects a VectorField")
    spatial = field._spatial_axis_indices()
    _validate_differentiation_axes(field, spatial, edge_order)
    values = np.moveaxis(field.values, field.component_axis, -1)
    spatial_rank = len(spatial)
    if values.shape[-1] != spatial_rank:
        raise ValidationError(
            f"Vector components ({values.shape[-1]}) must match spatial rank ({spatial_rank})"
        )
    unit = _single_spatial_unit(field, spatial)
    field_unit = cast(Unit, field.unit)
    result_unit = field_unit / unit if unit is not None else field_unit
    moved_index = {orig: index for index, orig in enumerate(spatial)}
    result = np.zeros(values.shape[:-1], dtype=np.result_type(values.dtype, float))
    for orig in spatial:
        moved_axis = moved_index[orig]
        result += np.gradient(  # type: ignore[call-overload]
            values[..., moved_axis],
            _spacing(field, orig),
            axis=moved_axis,
            edge_order=edge_order,
        )
    return ScalarField(
        values=result,
        unit=result_unit,
        coordinates=field.coordinates,
        coordinate_system=field.coordinate_system,
        metadata=field.metadata,
    )


@operator_registry.decorator("curl")
def curl(field: VectorField, *, edge_order: int = 1) -> VectorField:
    """Compute the curl of a 3D vector field."""
    if not isinstance(field, VectorField):
        raise ValidationError("curl expects a VectorField")
    spatial = field._spatial_axis_indices()
    _validate_differentiation_axes(field, spatial, edge_order)
    values = np.moveaxis(field.values, field.component_axis, -1)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValidationError("curl currently requires a 3D field with three components")
    if len(spatial) != 3:
        raise ValidationError("curl requires exactly three spatial dimensions")
    unit = _single_spatial_unit(field, spatial)
    field_unit = cast(Unit, field.unit)
    result_unit = field_unit / unit if unit is not None else field_unit
    dx = _spacing(field, spatial[0])
    dy = _spacing(field, spatial[1])
    dz = _spacing(field, spatial[2])
    vx, vy, vz = (values[..., index] for index in range(3))
    result = np.stack(
        [
            np.gradient(vz, dy, axis=1, edge_order=edge_order)  # type: ignore[call-overload]
            - np.gradient(vy, dz, axis=2, edge_order=edge_order),  # type: ignore[call-overload]
            np.gradient(vx, dz, axis=2, edge_order=edge_order)  # type: ignore[call-overload]
            - np.gradient(vz, dx, axis=0, edge_order=edge_order),  # type: ignore[call-overload]
            np.gradient(vy, dx, axis=0, edge_order=edge_order)  # type: ignore[call-overload]
            - np.gradient(vx, dy, axis=1, edge_order=edge_order),  # type: ignore[call-overload]
        ],
        axis=-1,
    )
    return VectorField(
        values=result,
        unit=result_unit,
        coordinates=field.coordinates,
        coordinate_system=field.coordinate_system,
        metadata=field.metadata,
    )


@operator_registry.decorator("laplacian")
def laplacian(field: ScalarField, *, edge_order: int = 1) -> ScalarField:
    """Compute a finite-difference scalar Laplacian."""
    if not isinstance(field, ScalarField):
        raise ValidationError("laplacian expects a ScalarField")
    spatial = field._spatial_axis_indices()
    _validate_differentiation_axes(field, spatial, edge_order)
    unit = _single_spatial_unit(field, spatial)
    field_unit = cast(Unit, field.unit)
    result_unit = field_unit / (unit**2) if unit is not None else field_unit
    result = np.zeros_like(field.values, dtype=np.result_type(field.values.dtype, float))
    for axis in spatial:
        spacing = _spacing(field, axis)
        first = np.gradient(  # type: ignore[call-overload]
            field.values, spacing, axis=axis, edge_order=edge_order
        )
        result += np.gradient(first, spacing, axis=axis, edge_order=edge_order)  # type: ignore[call-overload]
    return ScalarField(
        values=result,
        unit=result_unit,
        coordinates=field.coordinates,
        coordinate_system=field.coordinate_system,
        metadata=field.metadata,
    )


@operator_registry.decorator("integrate")
def integrate(
    values: Any, coordinates: Any | None = None, *, axis: int = -1
) -> Any:
    """Integrate samples with the composite trapezoidal rule.

    For a :class:`ScienceData` the coordinates default to the field's own coordinates
    and the result is a :class:`Quantity` whose unit is ``field.unit * coordinate.unit``.
    A plain ndarray is integrated and returned as an ndarray (no unit semantics).
    """
    if isinstance(values, ScienceData):
        field = values
        ndim = field.values.ndim
        axis = axis % ndim
        coords = coordinates if coordinates is not None else field.coordinates
        coord_array: np.ndarray | None = None
        coord_unit: Unit = units.parse("1")
        if coords is not None:
            spatial = field._spatial_axis_indices()
            if axis in spatial:
                coord_index = spatial.index(axis)
                if coord_index < len(coords):
                    coord_array = np.asarray(coords[coord_index], dtype=float)
                    if isinstance(field, (ScalarField, VectorField)):
                        coord_unit = _axis_unit(field, coord_index)
        result = np.trapezoid(field.values, x=coord_array, axis=axis)
        return Quantity(result, cast(Unit, field.unit) * coord_unit)
    array = np.asarray(values)
    return np.asarray(np.trapezoid(array, x=coordinates, axis=axis))


@operator_registry.decorator("fft")
def fft(
    values: Any,
    *,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> np.ndarray:
    """Compute a one-dimensional discrete Fourier transform (raw array, no units)."""
    array = values.values if hasattr(values, "values") else np.asarray(values)
    return np.fft.fft(array, axis=axis, norm=norm)


@operator_registry.decorator("ifft")
def ifft(
    values: Any,
    *,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> np.ndarray:
    """Compute an inverse one-dimensional discrete Fourier transform (raw array)."""
    array = values.values if hasattr(values, "values") else np.asarray(values)
    return np.fft.ifft(array, axis=axis, norm=norm)


@operator_registry.decorator("haar-wavelet")
def haar_wavelet(values: Any, *, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Return one-level Haar approximation and detail coefficients (raw arrays)."""
    array = np.asarray(values)
    moved = np.moveaxis(array, axis, -1)
    if moved.shape[-1] % 2:
        raise ValidationError("Haar wavelet input length must be even")
    left, right = moved[..., 0::2], moved[..., 1::2]
    scale = np.sqrt(2.0)
    return (left + right) / scale, (left - right) / scale


@operator_registry.decorator("convolution")
def convolution(
    values: Any,
    kernel: Any,
    *,
    mode: Literal["full", "same", "valid"] = "same",
) -> np.ndarray:
    """Compute one-dimensional convolution along the last axis."""
    array = np.asarray(values)
    weights = np.asarray(kernel)
    if weights.ndim != 1:
        raise ValidationError("The reference convolution kernel must be one-dimensional")
    return np.asarray(
        np.apply_along_axis(
            lambda row: np.convolve(row, weights, mode=mode), -1, array
        )
    )


@operator_registry.decorator("interpolate")
def interpolate(
    source_coordinates: Any,
    source_values: Any,
    target_coordinates: Any,
) -> np.ndarray:
    """Linearly interpolate one-dimensional scalar samples."""
    return np.asarray(np.interp(target_coordinates, source_coordinates, source_values))


@operator_registry.decorator("rotate-vectors")
def rotate_vectors(vectors: Any, rotation: Any) -> np.ndarray:
    """Apply an orthogonal transformation to vector components."""
    vector_array = np.asarray(vectors)
    matrix = np.asarray(rotation)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("Rotation must be a square matrix")
    if vector_array.shape[-1] != matrix.shape[0]:
        raise ValidationError("Vector dimension does not match rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(matrix.shape[0]), atol=1e-7):
        raise ValidationError("Rotation matrix must be orthogonal")
    return np.asarray(vector_array @ matrix.T)
