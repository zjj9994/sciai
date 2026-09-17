"""Unified reference training engine."""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from sciai.exceptions import MissingBackendError, ValidationError
from sciai.models import ScienceModel
from sciai.training.optim import Adam, Optimizer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 0
    precision: str = "float32"
    gradient_accumulation_steps: int = 1
    distributed_backend: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValidationError("epochs and batch_size must be positive")
        if self.precision not in {"float32", "float64"}:
            raise ValidationError("The unified trainer supports float32 and float64 precision")


@dataclass(frozen=True)
class TrainingState:
    epoch: int
    step: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


class Callback(Protocol):
    def on_train_begin(self, model: ScienceModel, config: TrainingConfig) -> None: ...

    def on_step_end(self, state: TrainingState) -> None: ...

    def on_train_end(self, state: TrainingState) -> None: ...


class ScienceTrainer:
    """Deterministic NumPy reference engine behind the stable trainer API."""

    trainer_type = "numpy"

    def __init__(
        self,
        *,
        config: TrainingConfig | None = None,
        optimizer: Optimizer | None = None,
        callbacks: Sequence[Callback] = (),
        **config_overrides: Any,
    ) -> None:
        if config is not None and config_overrides:
            raise ValidationError("Pass either TrainingConfig or keyword overrides, not both")
        self.config = config or TrainingConfig(**config_overrides)
        self.optimizer = optimizer or Adam()
        self.callbacks = tuple(callbacks)
        self.history: list[TrainingState] = []

    @staticmethod
    def _precision_dtype(precision: str) -> Any:
        return np.float32 if precision == "float32" else np.float64

    def fit(self, model: ScienceModel, dataset: Any) -> list[TrainingState]:
        if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise ValidationError("The reference trainer requires an indexable dataset")
        if self.config.distributed_backend is not None:
            raise MissingBackendError(
                "The NumPy reference trainer does not support distributed execution; "
                f"use a framework-specific trainer backend for {self.config.distributed_backend!r}"
            )
        if self.config.gradient_accumulation_steps < 1:
            raise ValidationError("gradient_accumulation_steps must be at least 1")
        accumulation_steps = self.config.gradient_accumulation_steps

        model.to_precision(self._precision_dtype(self.config.precision))

        parameters = model.trainable_parameters()
        if not parameters:
            raise MissingBackendError(
                f"{type(model).__name__} exposes no NumPy trainable parameters"
            )

        for callback in self.callbacks:
            callback.on_train_begin(model, self.config)

        self.history = []
        rng = np.random.default_rng(self.config.seed)
        previous_training = model.training
        model.training = True
        try:
            step = 0
            last_state: TrainingState | None = None
            pending_losses: list[float] = []
            pending_gradients: list[dict[str, np.ndarray]] = []
            pending_counts: list[int] = []
            for epoch in range(self.config.epochs):
                indices = np.arange(len(dataset))
                if self.config.shuffle:
                    rng.shuffle(indices)
                for start in range(0, len(indices), self.config.batch_size):
                    selected = indices[start : start + self.config.batch_size]
                    records = [dataset[int(index)] for index in selected]
                    if not records:
                        continue
                    batch = dataset.collate(records) if hasattr(dataset, "collate") else records
                    loss, gradients = model.loss_and_gradients(batch)
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
                    pending_losses.append(float(loss))
                    pending_gradients.append(dict(gradients))
                    pending_counts.append(len(records))
                    if len(pending_losses) >= accumulation_steps:
                        last_state = self._apply_accumulated(
                            parameters,
                            pending_losses,
                            pending_gradients,
                            pending_counts,
                            epoch,
                            step,
                        )
                        step += 1
                        pending_losses = []
                        pending_gradients = []
                        pending_counts = []
            if pending_losses:
                last_state = self._apply_accumulated(
                    parameters,
                    pending_losses,
                    pending_gradients,
                    pending_counts,
                    self.config.epochs - 1,
                    step,
                )
                step += 1

            if last_state is None:
                raise ValidationError("Cannot train on an empty dataset")
            for callback in self.callbacks:
                callback.on_train_end(last_state)
        finally:
            model.training = previous_training
        return list(self.history)

    def _apply_accumulated(
        self,
        parameters: MutableMapping[str, np.ndarray],
        losses: list[float],
        gradients_list: list[dict[str, np.ndarray]],
        counts: list[int],
        epoch: int,
        step: int,
    ) -> TrainingState:
        total = math.fsum(counts)
        accumulated_loss = 0.0
        accumulated_gradients: dict[str, np.ndarray] = {}
        for loss, gradients, count in zip(losses, gradients_list, counts, strict=True):
            weight = count / total
            accumulated_loss += weight * loss
            for name, gradient in gradients.items():
                contribution = gradient * weight
                if name in accumulated_gradients:
                    accumulated_gradients[name] = accumulated_gradients[name] + contribution
                else:
                    accumulated_gradients[name] = contribution
        accumulated_loss = float(accumulated_loss)
        if not np.isfinite(accumulated_loss):
            raise FloatingPointError(
                f"Non-finite accumulated loss at step {step}: {accumulated_loss}"
            )
        self.optimizer.step(parameters, accumulated_gradients)
        state = TrainingState(epoch=epoch, step=step, loss=accumulated_loss)
        self.history.append(state)
        for callback in self.callbacks:
            callback.on_step_end(state)
        return state
