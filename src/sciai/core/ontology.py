"""Scientific ontology primitives: dimensions, units, quantities, and frames."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

import numpy as np

from sciai.exceptions import DimensionError, ValidationError

BASE_DIMENSIONS = (
    "length",
    "mass",
    "time",
    "electric_current",
    "temperature",
    "amount_of_substance",
    "luminous_intensity",
)


@dataclass(frozen=True, init=False)
class Dimension:
    """A physical dimension represented by SI base-dimension exponents."""

    exponents: tuple[Fraction, ...]

    def __init__(
        self,
        exponents: dict[str, int | float | Fraction]
        | tuple[int | float | Fraction, ...]
        | None = None,
    ) -> None:
        if exponents is None:
            values = (Fraction(0),) * len(BASE_DIMENSIONS)
        elif isinstance(exponents, dict):
            unknown = set(exponents) - set(BASE_DIMENSIONS)
            if unknown:
                raise ValidationError(f"Unknown base dimensions: {sorted(unknown)}")
            values = tuple(Fraction(exponents.get(name, 0)) for name in BASE_DIMENSIONS)
        else:
            if len(exponents) != len(BASE_DIMENSIONS):
                raise ValidationError(
                    f"Expected {len(BASE_DIMENSIONS)} dimension exponents, got {len(exponents)}"
                )
            values = tuple(Fraction(value) for value in exponents)
        object.__setattr__(self, "exponents", values)

    @classmethod
    def dimensionless(cls) -> Dimension:
        return cls()

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents, strict=True)))

    def __pow__(self, power: int | Fraction) -> Dimension:
        exponent = Fraction(power)
        return Dimension(tuple(value * exponent for value in self.exponents))

    @property
    def is_dimensionless(self) -> bool:
        return all(value == 0 for value in self.exponents)

    def as_dict(self) -> dict[str, Fraction]:
        return {
            name: exponent
            for name, exponent in zip(BASE_DIMENSIONS, self.exponents, strict=True)
            if exponent
        }

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        return " ".join(f"{name}^{exponent}" for name, exponent in self.as_dict().items())


@dataclass(frozen=True)
class Unit:
    """A unit with an affine transform to its SI representation."""

    symbol: str
    dimension: Dimension = field(default_factory=Dimension.dimensionless)
    scale: float = 1.0
    offset: float = 0.0

    def _require_linear(self) -> None:
        if self.offset:
            raise DimensionError(f"Affine unit {self.symbol!r} cannot be composed")

    def __mul__(self, other: Unit) -> Unit:
        self._require_linear()
        other._require_linear()
        return Unit(
            f"{self.symbol}*{other.symbol}",
            self.dimension * other.dimension,
            self.scale * other.scale,
        )

    def __truediv__(self, other: Unit) -> Unit:
        self._require_linear()
        other._require_linear()
        return Unit(
            f"{self.symbol}/{other.symbol}",
            self.dimension / other.dimension,
            self.scale / other.scale,
        )

    def __pow__(self, power: int) -> Unit:
        self._require_linear()
        return Unit(f"{self.symbol}^{power}", self.dimension**power, self.scale**power)

    def convert_value(self, value: Any, target: Unit) -> np.ndarray:
        if self.dimension != target.dimension:
            raise DimensionError(
                f"Cannot convert {self.symbol} ({self.dimension}) to "
                f"{target.symbol} ({target.dimension})"
            )
        source: np.ndarray[Any, np.dtype[Any]] = np.asarray(value)
        si_value: np.ndarray[Any, np.dtype[Any]] = source * self.scale + self.offset
        result: np.ndarray[Any, np.dtype[Any]] = (si_value - target.offset) / target.scale
        return result


class UnitRegistry:
    """Registry and parser for a conservative subset of unit expressions."""

    _term_pattern = re.compile(r"^(?P<symbol>[A-Za-z]+)(?:\^(?P<power>-?\d+))?$")

    def __init__(self) -> None:
        self._units: dict[str, Unit] = {}

    def register(self, unit: Unit, *aliases: str) -> Unit:
        for key in (unit.symbol, *aliases):
            if key in self._units and self._units[key] != unit:
                raise ValidationError(f"Unit alias {key!r} is already registered")
            self._units[key] = unit
        return unit

    def get(self, symbol: str) -> Unit:
        try:
            return self._units[symbol]
        except KeyError as exc:
            raise ValidationError(f"Unknown unit {symbol!r}") from exc

    def parse(self, expression: str | Unit) -> Unit:
        if isinstance(expression, Unit):
            return expression
        compact = expression.strip().replace(" ", "").replace("**", "^")
        if compact in self._units:
            return self._units[compact]
        if not compact:
            raise ValidationError("Unit expression cannot be empty")
        # Reject malformed operator placement: leading, trailing, or consecutive.
        if compact[0] in "*/":
            raise ValidationError(
                f"Unit expression {expression!r} cannot begin with an operator"
            )
        if compact[-1] in "*/":
            raise ValidationError(
                f"Unit expression {expression!r} cannot end with an operator"
            )
        if re.search(r"[*/][*/]", compact):
            raise ValidationError(
                f"Unit expression {expression!r} has consecutive operators"
            )

        result = self.get("1")
        operation = "*"
        for token in re.split(r"([*/])", compact):
            if not token:
                continue
            if token in {"*", "/"}:
                operation = token
                continue
            match = self._term_pattern.fullmatch(token)
            if not match:
                raise ValidationError(
                    f"Unsupported unit expression {expression!r}; use products, quotients, "
                    "and integer powers (e.g. 'm/s^2')"
                )
            unit = self.get(match.group("symbol"))
            power = int(match.group("power") or 1)
            term = unit**power
            result = result * term if operation == "*" else result / term
        return Unit(compact, result.dimension, result.scale)

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._units))


LENGTH = Dimension({"length": 1})
MASS = Dimension({"mass": 1})
TIME = Dimension({"time": 1})
CURRENT = Dimension({"electric_current": 1})
TEMPERATURE = Dimension({"temperature": 1})
AMOUNT = Dimension({"amount_of_substance": 1})
LUMINOUS_INTENSITY = Dimension({"luminous_intensity": 1})

units = UnitRegistry()
units.register(Unit("1"), "dimensionless")
units.register(Unit("m", LENGTH))
units.register(Unit("km", LENGTH, 1_000.0))
units.register(Unit("cm", LENGTH, 0.01))
units.register(Unit("mm", LENGTH, 0.001))
units.register(Unit("angstrom", LENGTH, 1e-10))
units.register(Unit("kg", MASS))
units.register(Unit("g", MASS, 0.001))
units.register(Unit("s", TIME))
units.register(Unit("ms", TIME, 0.001))
units.register(Unit("min", TIME, 60.0))
units.register(Unit("h", TIME, 3_600.0))
units.register(Unit("A", CURRENT))
units.register(Unit("K", TEMPERATURE))
units.register(Unit("degC", TEMPERATURE, 1.0, 273.15))
units.register(Unit("mol", AMOUNT))
units.register(Unit("cd", LUMINOUS_INTENSITY))
units.register(Unit("rad"))
units.register(Unit("Hz", TIME**-1))
units.register(Unit("N", MASS * LENGTH / (TIME**2)))
units.register(Unit("Pa", MASS / LENGTH / (TIME**2)))
units.register(Unit("J", MASS * (LENGTH**2) / (TIME**2)))
units.register(Unit("W", MASS * (LENGTH**2) / (TIME**3)))
units.register(Unit("C", CURRENT * TIME))
units.register(Unit("V", MASS * (LENGTH**2) / (TIME**3) / CURRENT))
units.register(Unit("eV", MASS * (LENGTH**2) / (TIME**2), 1.602176634e-19))

# Derived units exposed for rate / differential calculus. ``parse`` already composes
# these, but registering them gives stable symbols and documents intent.
units.register(Unit("m/s", LENGTH / TIME), "velocity")
units.register(Unit("m/s^2", LENGTH / (TIME**2)), "acceleration")
units.register(Unit("1/m", LENGTH**-1), "wavenumber")
units.register(Unit("kg/(m*s)", MASS / LENGTH / TIME))
units.register(Unit("Pa*s", MASS / LENGTH / TIME))


@dataclass(frozen=True)
class Quantity:
    """A numerical value carrying a checked physical unit."""

    value: Any
    unit: Unit | str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", np.asarray(self.value))
        object.__setattr__(self, "unit", units.parse(self.unit))

    @property
    def dimension(self) -> Dimension:
        return cast(Unit, self.unit).dimension

    def to(self, target: Unit | str) -> Quantity:
        target_unit = units.parse(target)
        return Quantity(cast(Unit, self.unit).convert_value(self.value, target_unit), target_unit)

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        converted = other.to(self.unit)
        return Quantity(self.value + converted.value, self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        converted = other.to(self.unit)
        return Quantity(self.value - converted.value, self.unit)

    def __radd__(self, other: object) -> Quantity:
        return NotImplemented

    def __rsub__(self, other: object) -> Quantity:
        return NotImplemented

    def __mul__(self, other: Quantity | float | complex | np.ndarray) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(
                self.value * other.value,
                cast(Unit, self.unit) * cast(Unit, other.unit),
            )
        if isinstance(other, str):
            raise ValidationError("Quantity arithmetic does not accept string operands")
        if not isinstance(other, (int, float, complex, np.ndarray)):
            raise ValidationError(
                f"Quantity multiplication requires a numeric operand, got "
                f"{type(other).__name__}"
            )
        return Quantity(self.value * other, self.unit)

    def __rmul__(self, other: float | complex | np.ndarray) -> Quantity:
        if isinstance(other, str):
            raise ValidationError("Quantity arithmetic does not accept string operands")
        if not isinstance(other, (int, float, complex, np.ndarray)):
            return NotImplemented
        return Quantity(self.value * other, self.unit)

    def __truediv__(self, other: Quantity | float | complex | np.ndarray) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(
                self.value / other.value,
                cast(Unit, self.unit) / cast(Unit, other.unit),
            )
        if isinstance(other, str):
            raise ValidationError("Quantity arithmetic does not accept string operands")
        if not isinstance(other, (int, float, complex, np.ndarray)):
            raise ValidationError(
                f"Quantity division requires a numeric operand, got {type(other).__name__}"
            )
        return Quantity(self.value / other, self.unit)

    def __rtruediv__(self, other: float | complex | np.ndarray) -> Quantity:
        if isinstance(other, str):
            raise ValidationError("Quantity arithmetic does not accept string operands")
        if not isinstance(other, (int, float, complex, np.ndarray)):
            return NotImplemented
        unit = cast(Unit, self.unit)
        return Quantity(other / self.value, units.parse("1") / unit)


@dataclass(frozen=True)
class ReferenceFrame:
    """Named spatial reference frame with origin and orientation."""

    name: str = "world"
    origin: tuple[float, ...] = (0.0, 0.0, 0.0)
    orientation: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        origin: np.ndarray[Any, np.dtype[Any]] = np.asarray(self.origin, dtype=float)
        orientation: np.ndarray[Any, np.dtype[Any]] = np.asarray(self.orientation, dtype=float)
        if not np.all(np.isfinite(origin)):
            raise ValidationError("Reference-frame origin must be finite")
        if not np.all(np.isfinite(orientation)):
            raise ValidationError("Reference-frame orientation must be finite")
        if orientation.ndim != 2 or orientation.shape[0] != orientation.shape[1]:
            raise ValidationError("Reference-frame orientation must be a square matrix")
        if orientation.shape[0] != len(origin):
            raise ValidationError("Reference-frame orientation must match origin dimensions")
        identity = np.eye(orientation.shape[0])
        if not np.allclose(orientation @ orientation.T, identity, atol=1e-8):
            raise ValidationError("Reference-frame orientation must be orthogonal")
        determinant = float(np.linalg.det(orientation))
        if not np.isfinite(determinant) or abs(abs(determinant) - 1.0) > 1e-6:
            raise ValidationError(
                "Reference-frame orientation determinant must be approximately ±1"
            )
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(
            self, "orientation", tuple(tuple(float(value) for value in row) for row in orientation)
        )


@dataclass(frozen=True)
class CoordinateSystem:
    """Coordinate axes, axis units, handedness, and reference frame."""

    axes: tuple[str, ...] = ("x", "y", "z")
    axis_units: tuple[Unit | str, ...] = ("m", "m", "m")
    kind: str = "cartesian"
    handedness: str = "right"
    frame: ReferenceFrame = field(default_factory=ReferenceFrame)

    def __post_init__(self) -> None:
        if len(self.axes) == 0:
            raise ValidationError("Coordinate axes cannot be empty")
        if len(self.axes) != len(self.axis_units):
            raise ValidationError("Each coordinate axis must have one unit")
        if len(set(self.axes)) != len(self.axes):
            raise ValidationError("Coordinate axis names must be unique")
        if self.handedness not in {"left", "right"}:
            raise ValidationError("Coordinate handedness must be 'left' or 'right'")
        object.__setattr__(
            self, "axis_units", tuple(units.parse(unit) for unit in self.axis_units)
        )

    @property
    def axis_dimension(self) -> Dimension:
        return cast(Unit, self.axis_units[0]).dimension

    @property
    def length_based(self) -> bool:
        """True when every axis unit carries a length dimension (a metric frame)."""
        return all(cast(Unit, unit).dimension == LENGTH for unit in self.axis_units)

    @property
    def is_consistent(self) -> bool:
        """All axes share the same physical dimension (e.g. all length).

        A Cartesian metric frame should be length-based; frames that mix dimensions
        (e.g. spacetime ``(ct, x, y, z)``) expose this as ``False`` rather than being
        silently accepted as transformable. The numerical operators reject mixed units
        when they require a single propagated unit, avoiding non-existent transforms.
        """
        return all(cast(Unit, unit).dimension == self.axis_dimension for unit in self.axis_units)


@dataclass(frozen=True)
class ScienceMetadata:
    """Shared semantic metadata attached to scientific objects."""

    name: str = ""
    quantity: str = ""
    domain: str = "general"
    coordinate_system: CoordinateSystem | None = None
    reference_frame: ReferenceFrame | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
