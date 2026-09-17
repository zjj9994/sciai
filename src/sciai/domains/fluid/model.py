"""Fluid diagnostics behind the shared ScienceModel contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from sciai.domains.fluid.data import FlowField
from sciai.exceptions import ValidationError
from sciai.models import ScienceModel
from sciai.operators import divergence


class IncompressibilityModel(ScienceModel):
    """Evaluate the divergence residual of an incompressible flow field."""

    model_type = "incompressibility-residual"
    domain = "fluid"

    def __init__(self, *, reduction: str = "rms") -> None:
        if reduction not in {"none", "mean", "rms", "max"}:
            raise ValidationError(f"Unknown residual reduction {reduction!r}")
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def forward(self, inputs: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(f"Unexpected fluid model options: {sorted(kwargs)}")
        if not isinstance(inputs, FlowField):
            raise ValidationError("IncompressibilityModel expects a FlowField")
        residual = divergence(inputs).values
        if self.reduction == "none":
            # Raw divergence residual field (numpy array, no reduction applied).
            return residual
        if self.reduction == "mean":
            return float(np.mean(np.abs(residual)))
        if self.reduction == "max":
            return float(np.max(np.abs(residual)))
        return float(np.sqrt(np.mean(np.square(residual))))


class PINNSolver:
    """Stable PINN solver facade delegated to an installed backend plugin."""

    def __init__(self, *, physics: str, backend: Any | None = None) -> None:
        if not isinstance(physics, str) or not physics.strip():
            raise ValidationError("PINNSolver requires a non-empty physics description")
        self.physics = physics
        self.backend = backend

    def solve(
        self,
        flow: FlowField,
        *,
        boundary_conditions: Any,
        **kwargs: Any,
    ) -> FlowField:
        if not isinstance(flow, FlowField):
            raise ValidationError("PINNSolver.solve expects a FlowField")
        if self.backend is None:
            from sciai.exceptions import MissingBackendError

            raise MissingBackendError(
                "PINNSolver defines the stable fluid API, but numerical PINN training "
                "requires a framework plugin. Pass a backend implementing solve(flow, "
                "physics=..., boundary_conditions=...)."
            )
        result = self.backend.solve(
            flow,
            physics=self.physics,
            boundary_conditions=boundary_conditions,
            **kwargs,
        )
        if not isinstance(result, FlowField):
            raise ValidationError(
                f"PINNSolver backend for physics {self.physics!r} did not return a FlowField"
            )
        return result
