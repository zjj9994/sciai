"""Common mathematical and physical operators."""

from sciai.operators.numerical import (
    convolution,
    curl,
    divergence,
    fft,
    gradient,
    haar_wavelet,
    ifft,
    integrate,
    interpolate,
    laplacian,
    rotate_vectors,
)
from sciai.registry import operator_registry

__all__ = [
    "convolution",
    "curl",
    "divergence",
    "fft",
    "gradient",
    "haar_wavelet",
    "ifft",
    "integrate",
    "interpolate",
    "laplacian",
    "operator_registry",
    "rotate_vectors",
]
