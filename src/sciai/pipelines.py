"""High-level scientific task pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sciai.auto.factory import AutoModel
from sciai.auto.providers import PipelineProvider
from sciai.domains import load_builtin_plugins
from sciai.exceptions import ArtifactNotFoundError, RegistryError, ValidationError
from sciai.models import ScienceModel
from sciai.registry import pipeline_registry


class SciencePipeline(ABC):
    """Task-oriented preprocessing, model execution, and postprocessing."""

    task = "science-task"

    def __init__(self, model: ScienceModel) -> None:
        self.model = model

    def preprocess(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs

    @abstractmethod
    def postprocess(self, outputs: Any, **kwargs: Any) -> Any:
        """Convert model outputs into a task result."""

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        model_inputs = self.preprocess(inputs, **kwargs)
        outputs = self.model.predict(model_inputs, **kwargs)
        return self.postprocess(outputs, **kwargs)


def pipeline(
    task: str,
    *,
    model: str | ScienceModel,
    domain: str | None = None,
    **kwargs: Any,
) -> SciencePipeline:
    """Create a registered task pipeline, optionally disambiguated by domain."""
    load_builtin_plugins()
    if not isinstance(task, str) or not task.strip():
        raise ArtifactNotFoundError("A pipeline task must be a non-empty string")
    task = task.strip()
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        raise ArtifactNotFoundError("A pipeline domain must be a non-empty string")

    resolved_model = AutoModel.from_pretrained(model) if isinstance(model, str) else model
    if not isinstance(resolved_model, ScienceModel):
        raise ValidationError(
            f"Pipeline requires a ScienceModel, got {type(resolved_model).__name__}"
        )

    candidates = tuple(
        key for key in pipeline_registry if key == task or key.endswith(f"/{task}")
    )
    if domain is not None:
        domain_key = domain.strip().lower().replace("_", "-")
        candidates = tuple(key for key in candidates if key.startswith(f"{domain_key}/"))
    if not candidates:
        raise ArtifactNotFoundError(f"No pipeline is registered for task {task!r}")
    if len(candidates) > 1:
        model_domain = getattr(resolved_model, "domain", None)
        if model_domain:
            domain_key = str(model_domain).strip().lower().replace("_", "-")
            matching = tuple(key for key in candidates if key.startswith(f"{domain_key}/"))
            if len(matching) == 1:
                candidates = matching
            else:
                raise ArtifactNotFoundError(
                    f"Task {task!r} is available in multiple domains {candidates}; pass domain=..."
                )
        else:
            raise ArtifactNotFoundError(
                f"Task {task!r} is available in multiple domains {candidates}; pass domain=..."
            )
    try:
        provider = pipeline_registry.get(candidates[0])
    except RegistryError as exc:
        raise ArtifactNotFoundError(f"Pipeline {candidates[0]!r} disappeared") from exc
    factory = provider.factory if isinstance(provider, PipelineProvider) else provider
    result = factory(model=resolved_model, **kwargs)
    if not isinstance(result, SciencePipeline):
        raise ValidationError(
            f"Pipeline provider {candidates[0]!r} returned {type(result).__name__}, "
            "not a SciencePipeline"
        )
    return result
