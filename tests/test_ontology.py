import numpy as np
import pytest

from sciai import Dimension, Quantity, units
from sciai.exceptions import DimensionError, ValidationError


def test_quantity_conversion_and_arithmetic() -> None:
    distance = Quantity([1.0, 2.0], "km")
    converted = distance.to("m")
    np.testing.assert_allclose(converted.value, [1000.0, 2000.0])
    total = distance + Quantity([500.0, 500.0], "m")
    np.testing.assert_allclose(total.value, [1.5, 2.5])
    assert total.unit.symbol == "km"


def test_temperature_affine_conversion() -> None:
    freezing = Quantity(0.0, "degC").to("K")
    assert freezing.value == pytest.approx(273.15)


def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(DimensionError):
        Quantity(1.0, "m").to("s")


def test_unit_parser_and_dimension_algebra() -> None:
    velocity = units.parse("m/s")
    acceleration = units.parse("m/s^2")
    assert velocity.dimension == Dimension({"length": 1, "time": -1})
    assert acceleration.dimension == Dimension({"length": 1, "time": -2})


def test_unknown_units_are_rejected() -> None:
    with pytest.raises(ValidationError):
        units.parse("furlong")
