"""Framework-neutral model protocol and safe artifact persistence."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np

from sciai.exceptions import MissingBackendError, ValidationError

ModelT = TypeVar("ModelT", bound="ScienceModel")


class ScienceModel(ABC):
    """Stable model interface independent of any tensor framework.

    Domain models implement :meth:`forward`. Trainable reference models may also
    implement ``loss_and_gradients`` and ``trainable_parameters`` for the NumPy
    trainer. Framework integrations can provide specialized trainers without
    changing this public model contract.
    """

    model_type = "science-model"

    def __init__(self, **config: Any) -> None:
        self.config: dict[str, Any] = dict(config)
        self.training = False

    @abstractmethod
    def forward(self, inputs: Any, **kwargs: Any) -> Any:
        """Run the model's scientific computation."""

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        return self.forward(inputs, **kwargs)

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Run inference while preserving the previous training mode."""
        previous = self.training
        self.training = False
        try:
            return self.forward(inputs, **kwargs)
        finally:
            self.training = previous

    def train(
        self,
        dataset: Any | None = None,
        *,
        trainer: Any | None = None,
        **trainer_kwargs: Any,
    ) -> ScienceModel | Any:
        """Enter training mode or fit against a dataset through a trainer."""
        self.training = True
        if dataset is None:
            return self
        if trainer is None:
            from sciai.training import ScienceTrainer

            trainer = ScienceTrainer(**trainer_kwargs)
        return trainer.fit(self, dataset)

    def eval(self) -> ScienceModel:
        self.training = False
        return self

    def trainable_parameters(self) -> MutableMapping[str, np.ndarray]:
        """Return mutable NumPy parameters for the reference trainer."""
        return {}

    def to_precision(self, dtype: Any) -> ScienceModel:
        """Validate a requested precision; models may override to cast their state.

        The base class cannot safely replace arrays exposed through an arbitrary
        ``trainable_parameters()`` mapping, so it preserves existing storage. Models
        that own NumPy arrays directly override this method to perform the cast.
        """
        target = np.dtype(dtype)
        if target.kind not in {"f", "c"}:
            raise ValidationError("Model precision must be a floating or complex NumPy dtype")
        return self

    def loss_and_gradients(self, batch: Any) -> tuple[float, Mapping[str, np.ndarray]]:
        """Compute a scalar loss and gradients for the reference trainer."""
        raise MissingBackendError(
            f"{type(self).__name__} does not implement NumPy loss/gradient computation; "
            "use a domain-specific trainer backend"
        )

    def state_dict(self) -> Mapping[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in self.trainable_parameters().items()}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        parameters = self.trainable_parameters()
        unknown = set(state) - set(parameters)
        missing = set(parameters) - set(state)
        if unknown or missing:
            raise ValidationError(
                f"State mismatch for {type(self).__name__}: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name, value in state.items():
            target = parameters[name]
            array = np.asarray(value)
            if target.shape != array.shape:
                raise ValidationError(
                    f"Parameter {name!r} shape mismatch: expected {target.shape}, got {array.shape}"
                )
            target[...] = array

    def save(self, path: str | Path) -> Path:
        """Save configuration and NumPy weights without executable pickle data."""
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": 1,
            "model_type": self.model_type,
            "class_name": type(self).__name__,
            "config": self.config,
        }
        (destination / "model.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        weights = dict(self.state_dict())
        # NumPy's stubs cannot express dynamically named arrays passed as **kwargs.
        save_arrays = cast(Any, np.savez_compressed)
        save_arrays(destination / "weights.npz", **weights)
        return destination

    save_pretrained = save

    @classmethod
    def from_config(cls: type[ModelT], config: Mapping[str, Any]) -> ModelT:
        return cls(**dict(config))

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        source = Path(path)
        manifest_path = source / "model.json"
        weights_path = source / "weights.npz"
        if not manifest_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Incomplete sciai model artifact at {source}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise ValidationError(
                f"Unsupported model artifact version {manifest.get('format_version')!r}"
            )
        if cls is not ScienceModel and manifest.get("model_type") != cls.model_type:
            raise ValidationError(
                f"Artifact model type {manifest.get('model_type')!r} does not match "
                f"{cls.model_type!r}"
            )
        model = cls.from_config(manifest.get("config", {}))
        with np.load(weights_path, allow_pickle=False) as archive:
            model.load_state_dict({name: archive[name] for name in archive.files})
        return model

    load_pretrained = load
