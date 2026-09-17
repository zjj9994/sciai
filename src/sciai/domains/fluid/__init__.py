"""Built-in fluid-science reference plugin."""

from sciai.auto import ModelProvider, PipelineProvider
from sciai.domains.base import DomainPlugin
from sciai.domains.fluid.data import FlowField
from sciai.domains.fluid.model import IncompressibilityModel, PINNSolver
from sciai.domains.fluid.pipeline import FlowResidualPipeline

plugin = DomainPlugin(
    name="fluid",
    version="0.1.0",
    description="Fluid field semantics and conservation diagnostics.",
    subdomains=("fluid-mechanics", "aerodynamics", "hydrodynamics"),
    data_types=(FlowField,),
    models={
        "incompressibility-residual": ModelProvider(
            factory=IncompressibilityModel,
            domain="fluid",
            tasks=("conservation-check",),
            description="Finite-difference divergence diagnostic.",
        )
    },
    pipelines={
        "fluid/conservation-check": PipelineProvider(
            factory=FlowResidualPipeline,
            domains=("fluid",),
        )
    },
    metadata={
        "production_integrations": ("OpenFOAM", "Fluent", "FEniCS", "PyTorch", "JAX"),
        "integration_status": "plugin-ready",
    },
)

__all__ = [
    "FlowField",
    "FlowResidualPipeline",
    "IncompressibilityModel",
    "PINNSolver",
    "plugin",
]
