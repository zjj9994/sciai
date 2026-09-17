"""Fluid task pipelines."""

from __future__ import annotations

from typing import Any

from sciai.pipelines import SciencePipeline


class FlowResidualPipeline(SciencePipeline):
    task = "conservation-check"

    def postprocess(self, outputs: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "metric": "incompressibility_residual_rms",
            "value": float(outputs),
            "unit": "1/s",
            "model": self.model.model_type,
        }
