"""Exception hierarchy for sciai."""


class SciAIError(Exception):
    """Base exception for all sciai errors."""


class ValidationError(SciAIError, ValueError):
    """Raised when scientific data violates a declared invariant."""


class DimensionError(ValidationError):
    """Raised when incompatible physical dimensions are combined."""


class RegistryError(SciAIError, LookupError):
    """Raised for missing or conflicting registry entries."""


class MissingBackendError(SciAIError, RuntimeError):
    """Raised when an optional execution backend is required."""


class ArtifactNotFoundError(SciAIError, FileNotFoundError):
    """Raised when a model or dataset identifier cannot be resolved."""


class CouplingError(SciAIError, RuntimeError):
    """Raised when a multiphysics coupling contract cannot be satisfied."""
