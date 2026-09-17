import numpy as np
import pytest

from sciai.io import load_data
from sciai.metrics import MAE, R2, RMSE, RelativeError
from sciai.scaffold import scaffold_domain


def test_regression_metrics() -> None:
    prediction = [1.0, 2.0, 4.0]
    target = [1.0, 3.0, 3.0]
    metrics = [MAE(), RMSE(), R2(), RelativeError()]
    for metric in metrics:
        metric.update(prediction, target)
        assert np.isfinite(metric.compute())


def test_npy_loading(tmp_path) -> None:
    path = tmp_path / "field.npy"
    np.save(path, np.arange(6).reshape(2, 3))
    field = load_data(path, unit="K")
    assert field.shape == (2, 3)
    assert field.unit.symbol == "K"


def test_scaffold_domain(tmp_path) -> None:
    result = scaffold_domain("plasma-physics", tmp_path)
    assert (result / "pyproject.toml").is_file()
    assert (result / "src" / "sciai_plasma_physics" / "__init__.py").is_file()
    with pytest.raises(FileExistsError):
        scaffold_domain("plasma-physics", tmp_path)
