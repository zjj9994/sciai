import numpy as np

from sciai import ScalarField, VectorField
from sciai.operators import curl, divergence, gradient, haar_wavelet, laplacian


def test_gradient_and_laplacian() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = ScalarField(values=xx**2 + yy**2, coordinates=(x, y))
    grad = gradient(field, edge_order=2)
    lap = laplacian(field, edge_order=2)
    np.testing.assert_allclose(grad.values[5:-5, 5:-5, 0], 2 * xx[5:-5, 5:-5], atol=1e-8)
    np.testing.assert_allclose(lap.values[5:-5, 5:-5], 4.0, atol=1e-8)


def test_divergence() -> None:
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.stack([xx, -yy], axis=-1)
    field = VectorField(values=values, coordinates=(x, y))
    np.testing.assert_allclose(divergence(field, edge_order=2).values, 0.0, atol=1e-8)


def test_curl_of_rigid_rotation() -> None:
    axis = np.linspace(-1.0, 1.0, 7)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.stack([-y, x, np.zeros_like(z)], axis=-1)
    field = VectorField(values=values, coordinates=(axis, axis, axis))
    result = curl(field, edge_order=2)
    np.testing.assert_allclose(result.values[..., :2], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.values[..., 2], 2.0, atol=1e-8)


def test_haar_wavelet() -> None:
    approximation, detail = haar_wavelet([1.0, 1.0, 2.0, 0.0])
    np.testing.assert_allclose(approximation, [np.sqrt(2.0), np.sqrt(2.0)])
    np.testing.assert_allclose(detail, [0.0, np.sqrt(2.0)])
