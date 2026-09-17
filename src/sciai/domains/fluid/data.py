"""Fluid-domain field containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sciai.core import ScalarField, ScienceMetadata, VectorField
from sciai.exceptions import ValidationError


def _coord_index(name: str) -> int:
    suffix = name[len("coord_") :]
    if not suffix.isdigit():
        raise ValidationError(f"Fluid NPZ coordinate {name!r} must be 'coord_<int>'")
    return int(suffix)


@dataclass(kw_only=True)
class FlowField(VectorField):
    """Velocity field with optional pressure and fluid properties."""

    pressure: ScalarField | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        spatial_shape = (
            self.values.shape[: self.component_axis]
            + self.values.shape[self.component_axis + 1 :]
        )
        if self.pressure is not None and self.pressure.shape != spatial_shape:
            raise ValidationError(
                f"Pressure shape {self.pressure.shape} does not match velocity grid "
                f"{spatial_shape}"
            )

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray(self.values)

    @classmethod
    def from_arrays(
        cls,
        velocity: Any,
        *,
        pressure: Any | None = None,
        coordinates: tuple[Any, ...] | None = None,
        velocity_unit: str = "m/s",
        pressure_unit: str = "Pa",
        component_axis: int = -1,
        properties: dict[str, Any] | None = None,
    ) -> FlowField:
        pressure_field = (
            ScalarField(
                values=pressure,
                unit=pressure_unit,
                coordinates=coordinates,
                metadata=ScienceMetadata(quantity="pressure", domain="fluid"),
            )
            if pressure is not None
            else None
        )
        return cls(
            values=velocity,
            unit=velocity_unit,
            coordinates=coordinates,
            component_axis=component_axis,
            pressure=pressure_field,
            properties=properties or {},
            metadata=ScienceMetadata(quantity="velocity", domain="fluid"),
        )

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> FlowField:
        """Load the core NPZ interchange format.

        The archive must contain ``velocity`` and may contain ``pressure`` and
        ``coord_0``, ``coord_1``, ... arrays, consecutively indexed from zero.
        VTK/OpenFOAM readers are separate optional plugins so the core package
        remains lightweight.
        """
        source = Path(path)
        if source.suffix.lower() != ".npz":
            raise ValidationError(
                "The core fluid loader supports .npz. Install a VTK/OpenFOAM loader "
                "plugin for native solver files."
            )
        try:
            with np.load(source, allow_pickle=False) as archive:
                if "velocity" not in archive.files:
                    raise ValidationError("Fluid NPZ requires a 'velocity' array")
                coord_keys = sorted(
                    (key for key in archive.files if key.startswith("coord_")),
                    key=_coord_index,
                )
                indices = [_coord_index(name) for name in coord_keys]
                if indices != list(range(len(indices))):
                    raise ValidationError(
                        "Fluid NPZ coordinate arrays must be consecutively indexed as "
                        f"coord_0, coord_1, ...; found {coord_keys or '<none>'}"
                    )
                coordinates = tuple(archive[name] for name in coord_keys)
                pressure = archive.get("pressure", None)
                velocity = archive["velocity"]
        except ValueError as exc:
            raise ValidationError(f"Could not load fluid NPZ {source}: {exc}") from exc
        return cls.from_arrays(
            velocity,
            pressure=pressure,
            coordinates=coordinates or None,
            **kwargs,
        )
