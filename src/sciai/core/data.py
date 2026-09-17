"""The eight domain-neutral scientific data structures."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from sciai.core.ontology import CoordinateSystem, ScienceMetadata, Unit, units
from sciai.exceptions import ValidationError


def _finite_no_inf(array: np.ndarray) -> bool:
    """True when the array contains no infinite values (NaNs are legitimate missing data)."""
    return not np.any(np.isinf(np.asarray(array)))


@dataclass(kw_only=True)
class ScienceData:
    """Base value object shared by every scientific data representation."""

    values: Any
    unit: Unit | str = "1"
    coordinates: tuple[Any, ...] | None = None
    coordinate_system: CoordinateSystem | None = None
    metadata: ScienceMetadata = field(default_factory=ScienceMetadata)

    def _spatial_axis_indices(self) -> list[int]:
        """Value-dimension indices that carry coordinates (all dims by default)."""
        return list(range(self.values.ndim))

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.unit = units.parse(self.unit)
        if self.values.size == 0:
            raise ValidationError(f"{type(self).__name__} values cannot be empty")
        if self.coordinates is not None:
            self.coordinates = tuple(np.asarray(axis) for axis in self.coordinates)
            spatial = self._spatial_axis_indices()
            if len(self.coordinates) != len(spatial):
                raise ValidationError(
                    f"{type(self).__name__} expects {len(spatial)} coordinate axes, "
                    f"got {len(self.coordinates)}"
                )
            for index, axis_index in enumerate(spatial):
                axis = self.coordinates[index]
                if axis.ndim != 1:
                    raise ValidationError("Coordinate axes must be one-dimensional")
                if len(axis) != self.values.shape[axis_index]:
                    raise ValidationError(
                        f"Coordinate axis {index} has length {len(axis)}, expected "
                        f"{self.values.shape[axis_index]}"
                    )
        if (
            self.coordinate_system is not None
            and self.coordinates is not None
            and len(self.coordinates) != len(self.coordinate_system.axes)
        ):
            raise ValidationError(
                "Coordinate count does not match the coordinate system axes "
                f"({len(self.coordinates)} vs {len(self.coordinate_system.axes)})"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self.values.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return cast(np.dtype[Any], np.dtype(self.values.dtype))

    def to(self, target_unit: Unit | str) -> ScienceData:
        """Convert to ``target_unit`` while preserving every subtype field."""
        parsed = units.parse(target_unit)
        unit = cast(Unit, self.unit)
        converted = unit.convert_value(self.values, parsed)
        return replace(self, values=converted, unit=parsed)

    def summary(self) -> dict[str, Any]:
        unit = cast(Unit, self.unit)
        return {
            "type": type(self).__name__,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "unit": unit.symbol,
            "domain": self.metadata.domain,
        }


@dataclass(kw_only=True)
class ScalarField(ScienceData):
    """A scalar value sampled over one or more spatial dimensions."""


@dataclass(kw_only=True)
class VectorField(ScienceData):
    """A vector-valued field with an explicit component axis."""

    component_axis: int = -1

    def _spatial_axis_indices(self) -> list[int]:
        return [i for i in range(self.values.ndim) if i != self.component_axis]

    def __post_init__(self) -> None:
        if self.values.ndim < 2:
            raise ValidationError("A vector field needs spatial and component dimensions")
        axis = self.component_axis % self.values.ndim
        if self.values.shape[axis] < 1:
            raise ValidationError("Vector fields need at least one component")
        self.component_axis = axis
        super().__post_init__()


@dataclass(kw_only=True)
class TensorField(ScienceData):
    """A tensor-valued field with trailing tensor dimensions."""

    tensor_rank: int = 2

    def _spatial_axis_indices(self) -> list[int]:
        return list(range(self.values.ndim - self.tensor_rank))

    def __post_init__(self) -> None:
        if self.tensor_rank < 2:
            raise ValidationError("TensorField tensor_rank must be at least 2")
        if self.values.ndim <= self.tensor_rank:
            raise ValidationError("A tensor field needs spatial and tensor dimensions")
        super().__post_init__()


@dataclass(kw_only=True)
class GraphData(ScienceData):
    """A graph represented by node features and a canonical (2, E) edge index."""

    edge_index: Any
    edge_features: Any | None = None
    node_positions: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        if self.values.ndim not in (1, 2):
            raise ValidationError("Graph node features must be one- or two-dimensional")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValidationError("Graph edge_index must have shape (2, edges)")
        edge_count = self.edge_index.shape[1]
        if self.edge_index.size and (
            self.edge_index.min() < 0 or self.edge_index.max() >= self.values.shape[0]
        ):
            raise ValidationError("Graph edges reference an unknown node")
        if self.edge_features is not None:
            self.edge_features = np.asarray(self.edge_features)
            if self.edge_features.ndim != 2:
                raise ValidationError("Edge features must be a 2-D (edges, channels) array")
            if self.edge_features.shape[0] != edge_count:
                raise ValidationError("There must be one edge-feature row per edge")
        if self.node_positions is not None:
            self.node_positions = np.asarray(self.node_positions)
            if self.node_positions.shape[0] != self.values.shape[0]:
                raise ValidationError("There must be one position per graph node")

    @property
    def node_features(self) -> np.ndarray:
        return np.asarray(self.values)


@dataclass(kw_only=True)
class ParticleSystem(ScienceData):
    """Particle positions with optional per-particle attributes."""

    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.values.ndim != 2 or self.values.shape[1] not in {1, 2, 3}:
            raise ValidationError("Particle positions must have shape (particles, 1|2|3)")
        # Infinite values are non-physical; NaN is permitted as a missing-value marker.
        if not _finite_no_inf(self.values):
            raise ValidationError("Particle positions must not contain infinite values")
        particle_count = self.values.shape[0]
        converted: dict[str, np.ndarray] = {}
        for name, value in self.attributes.items():
            array = np.asarray(value)
            if len(array) != particle_count:
                raise ValidationError(f"Particle attribute {name!r} has the wrong length")
            converted[name] = array
        self.attributes = converted

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.values)


@dataclass(kw_only=True)
class MeshData(ScienceData):
    """Unstructured mesh points and typed cell-connectivity arrays."""

    cells: dict[str, Any]
    point_data: dict[str, Any] = field(default_factory=dict)
    cell_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.values.ndim != 2 or self.values.shape[1] not in {1, 2, 3}:
            raise ValidationError("Mesh points must have shape (points, 1|2|3)")
        if not _finite_no_inf(self.values):
            raise ValidationError("Mesh points must not contain infinite values")
        point_count = self.values.shape[0]
        normalized_cells: dict[str, np.ndarray] = {}
        for name, connectivity in self.cells.items():
            array = np.asarray(connectivity, dtype=np.int64)
            if array.ndim != 2:
                raise ValidationError(f"Mesh cell block {name!r} must be two-dimensional")
            if array.size and (array.min() < 0 or array.max() >= point_count):
                raise ValidationError(f"Mesh cell block {name!r} references an unknown point")
            normalized_cells[name] = array
        self.cells = normalized_cells
        self.point_data = {name: np.asarray(value) for name, value in self.point_data.items()}
        if any(len(value) != point_count for value in self.point_data.values()):
            raise ValidationError("Each point-data array must align with mesh points")

    @property
    def points(self) -> np.ndarray:
        return np.asarray(self.values)


@dataclass(kw_only=True)
class TimeSeries(ScienceData):
    """Values indexed by a strictly increasing time coordinate."""

    time: Any
    time_unit: Unit | str = "s"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.time = np.asarray(self.time)
        self.time_unit = units.parse(self.time_unit)
        if self.time.ndim != 1 or len(self.time) != self.values.shape[0]:
            raise ValidationError("Time coordinate must align with the first value dimension")
        if np.any(np.isinf(self.time)):
            raise ValidationError("Time coordinate must not contain infinite values")
        if len(self.time) > 1 and np.any(np.diff(self.time) <= 0):
            raise ValidationError("Time coordinates must be strictly increasing")


@dataclass(kw_only=True)
class Spectrum(ScienceData):
    """Spectral intensity indexed by frequency, wavelength, energy, or wavenumber."""

    axis: Any
    axis_unit: Unit | str = "Hz"
    axis_kind: str = "frequency"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.axis = np.asarray(self.axis)
        self.axis_unit = units.parse(self.axis_unit)
        if self.axis.ndim != 1 or len(self.axis) != self.values.shape[0]:
            raise ValidationError("Spectrum axis must align with the first value dimension")
        if np.any(np.isinf(self.axis)):
            raise ValidationError("Spectrum axis must not contain infinite values")
        if len(self.axis) > 1 and not (
            np.all(np.diff(self.axis) > 0) or np.all(np.diff(self.axis) < 0)
        ):
            raise ValidationError("Spectrum axis must be strictly monotonic")
        if self.axis_kind not in {"frequency", "wavelength", "energy", "wavenumber"}:
            raise ValidationError(f"Unsupported spectrum axis kind {self.axis_kind!r}")
