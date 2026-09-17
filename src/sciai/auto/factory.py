"""Auto factories for local artifacts and registered domain providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sciai.auto.providers import DatasetProvider, ModelProvider
from sciai.domains import load_builtin_plugins
from sciai.exceptions import ArtifactNotFoundError, RegistryError, ValidationError
from sciai.models import ScienceModel
from sciai.registry import dataset_registry, model_registry, trainer_registry
from sciai.training import ScienceTrainer, TrainingConfig


def _identifier_key(identifier: str) -> str:
    value = identifier.strip()
    if value.startswith("sciai/"):
        value = value[len("sciai/") :]
    return value


class AutoModel:
    """Resolve a model from a local artifact or an installed provider."""

    @classmethod
    def from_pretrained(cls, identifier: str | Path, **kwargs: Any) -> ScienceModel:
        load_builtin_plugins()
        if isinstance(identifier, (str, Path)):
            path = Path(identifier).expanduser()
            if path.is_dir():
                return cls._from_local_artifact(path, **kwargs)
            if path.is_file():
                raise ArtifactNotFoundError(
                    f"{path} is a file; AutoModel.from_pretrained expects a directory artifact"
                )

        key = _identifier_key(str(identifier))
        try:
            provider = model_registry.get(key)
        except RegistryError as exc:
            raise ArtifactNotFoundError(
                f"Model {identifier!r} is not local or registered. Installed models: "
                f"{', '.join(model_registry.keys()) or '<none>'}"
            ) from exc
        if isinstance(provider, ModelProvider):
            return provider.factory(**kwargs)
        if isinstance(provider, type) and issubclass(provider, ScienceModel):
            return provider(**kwargs)
        if callable(provider):
            model = provider(**kwargs)
            if isinstance(model, ScienceModel):
                return model
        raise ValidationError(f"Registered model provider {key!r} is invalid")

    @classmethod
    def _from_local_artifact(cls, path: Path, **kwargs: Any) -> ScienceModel:
        manifest_path = path / "model.json"
        if not manifest_path.is_file():
            raise ArtifactNotFoundError(f"No model.json found in {path}")
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON in {manifest_path}: {exc}") from exc
        except OSError as exc:
            raise ValidationError(f"Could not read {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValidationError(
                f"Model manifest at {path} must be a JSON object, got {type(manifest).__name__}"
            )
        model_type = manifest.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            raise ValidationError(
                f"Model manifest at {path} must declare a non-empty string 'model_type'"
            )
        config = manifest.get("config", {})
        if not isinstance(config, dict):
            raise ValidationError(
                f"Model manifest at {path} 'config' must be a JSON object"
            )
        try:
            provider = model_registry.get(model_type)
        except RegistryError as exc:
            raise ArtifactNotFoundError(
                f"No installed plugin can load model type {model_type!r}"
            ) from exc
        model_class = provider.factory if isinstance(provider, ModelProvider) else provider
        if not isinstance(model_class, type) or not issubclass(model_class, ScienceModel):
            raise ValidationError(
                f"Model provider {model_type!r} does not expose a loadable model class"
            )
        return model_class.load(path)


class AutoDataset:
    """Resolve a standardized dataset from an installed provider."""

    @classmethod
    def load(cls, identifier: str, **kwargs: Any) -> Any:
        load_builtin_plugins()
        key = _identifier_key(identifier)
        try:
            provider = dataset_registry.get(key)
        except RegistryError as exc:
            raise ArtifactNotFoundError(
                f"Dataset {identifier!r} is not registered. Installed datasets: "
                f"{', '.join(dataset_registry.keys()) or '<none>'}"
            ) from exc
        if isinstance(provider, DatasetProvider):
            dataset = provider.factory(**kwargs)
        elif callable(provider):
            dataset = provider(**kwargs)
        else:
            raise ValidationError(
                f"Registered dataset provider {key!r} is neither a DatasetProvider nor callable"
            )
        if dataset is None:
            raise ValidationError(
                f"Dataset provider {key!r} returned nothing usable"
            )
        return dataset


class AutoTrainer:
    """Build a trainer backend from a shared configuration."""

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig | Mapping[str, Any] | None = None,
        *,
        backend: str = "numpy",
        **kwargs: Any,
    ) -> ScienceTrainer:
        load_builtin_plugins()
        if backend == "numpy":
            trainer_type: Any = ScienceTrainer
        else:
            try:
                trainer_type = trainer_registry.get(backend)
            except RegistryError as exc:
                raise ArtifactNotFoundError(
                    f"Trainer backend {backend!r} is not installed"
                ) from exc
        if not (isinstance(trainer_type, type) and issubclass(trainer_type, ScienceTrainer)):
            raise ValidationError(
                f"Trainer provider {backend!r} does not expose a ScienceTrainer subclass"
            )
        normalized = (
            config
            if isinstance(config, TrainingConfig)
            else TrainingConfig(**dict(config or {}))
        )
        return trainer_type(config=normalized, **kwargs)
