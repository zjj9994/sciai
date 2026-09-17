"""Small NumPy reference optimizers and scientific learning-rate schedules."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sciai.exceptions import ValidationError


def _require_float(value: Any, name: str) -> float:
    """Reject integer/boolean scalars so optimizers never silently widen them."""
    if isinstance(value, (bool, int)):
        raise ValidationError(f"{name} must be a float, not an integer")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return result


def _require_beta(value: Any, name: str) -> float:
    result = _require_float(value, name)
    if not 0.0 <= result < 1.0:
        raise ValidationError(f"{name} must be in [0, 1), got {result!r}")
    return result


def _require_non_negative(value: Any, name: str) -> float:
    result = _require_float(value, name)
    if result < 0.0:
        raise ValidationError(f"{name} must be non-negative, got {result!r}")
    return result


def _require_positive(value: Any, name: str) -> float:
    result = _require_float(value, name)
    if result <= 0.0:
        raise ValidationError(f"{name} must be positive, got {result!r}")
    return result


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValidationError(f"{name} must be at least 1, got {value!r}")
    return value


def _is_inexact_array(array: Any) -> bool:
    return isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.inexact)


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, step: int) -> float:
        """Return the learning rate for a zero-based optimizer step."""


@dataclass(frozen=True)
class ConstantLR(LearningRateSchedule):
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        learning_rate = _require_positive(self.learning_rate, "learning_rate")
        object.__setattr__(self, "learning_rate", learning_rate)

    def __call__(self, step: int) -> float:
        del step
        return self.learning_rate


@dataclass(frozen=True)
class CosineDecayLR(LearningRateSchedule):
    initial_rate: float = 1e-3
    total_steps: int = 1_000
    minimum_rate: float = 0.0

    def __post_init__(self) -> None:
        initial_rate = _require_positive(self.initial_rate, "initial_rate")
        minimum_rate = _require_float(self.minimum_rate, "minimum_rate")
        total_steps = _require_positive_int(self.total_steps, "total_steps")
        object.__setattr__(self, "initial_rate", initial_rate)
        object.__setattr__(self, "minimum_rate", minimum_rate)
        object.__setattr__(self, "total_steps", total_steps)

    def __call__(self, step: int) -> float:
        progress = min(max(step, 0), self.total_steps) / max(self.total_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_rate + (self.initial_rate - self.minimum_rate) * cosine


@dataclass(frozen=True)
class WarmupCosineLR(LearningRateSchedule):
    peak_rate: float = 1e-3
    warmup_steps: int = 100
    total_steps: int = 1_000
    minimum_rate: float = 0.0

    def __post_init__(self) -> None:
        peak_rate = _require_positive(self.peak_rate, "peak_rate")
        minimum_rate = _require_float(self.minimum_rate, "minimum_rate")
        warmup_steps = _require_positive_int(self.warmup_steps, "warmup_steps")
        total_steps = _require_positive_int(self.total_steps, "total_steps")
        object.__setattr__(self, "peak_rate", peak_rate)
        object.__setattr__(self, "minimum_rate", minimum_rate)
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "total_steps", total_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_rate * (step + 1) / max(self.warmup_steps, 1)
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min(step - self.warmup_steps, decay_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_rate + (self.peak_rate - self.minimum_rate) * cosine


class Optimizer(ABC):
    def __init__(self, learning_rate: float | LearningRateSchedule = 1e-3) -> None:
        if isinstance(learning_rate, bool):
            raise ValidationError("learning_rate must be a float, not an integer")
        if isinstance(learning_rate, LearningRateSchedule):
            self.schedule = learning_rate
        elif isinstance(learning_rate, int):
            raise ValidationError("learning_rate must be a float, not an integer")
        else:
            self.schedule = ConstantLR(float(learning_rate))
        self.step_count = 0
        self._validate_schedule()

    def _validate_schedule(self) -> None:
        rate = self.schedule(0)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValidationError(
                f"Learning rate schedule returned invalid rate {rate!r} at step 0"
            )

    @abstractmethod
    def step(
        self,
        parameters: MutableMapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        """Update parameters in place."""

    def _validate(
        self,
        parameters: Mapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        if set(parameters) != set(gradients):
            raise ValidationError(
                f"Gradient keys must match parameters: "
                f"{sorted(gradients)} != {sorted(parameters)}"
            )
        for name in parameters:
            parameter = parameters[name]
            gradient = gradients[name]
            if not _is_inexact_array(parameter):
                raise ValidationError(
                    f"Parameter {name!r} must be a floating/complex array, "
                    f"got {getattr(parameter, 'dtype', type(parameter))!r}"
                )
            if not _is_inexact_array(gradient):
                raise ValidationError(
                    f"Gradient {name!r} must be a floating/complex array, "
                    f"got {getattr(gradient, 'dtype', type(gradient))!r}"
                )
            if parameter.shape != gradient.shape:
                raise ValidationError(f"Gradient shape mismatch for parameter {name!r}")
            if not np.all(np.isfinite(parameter)):
                raise ValidationError(f"Parameter {name!r} is not finite")
            if not np.all(np.isfinite(gradient)):
                raise ValidationError(f"Gradient {name!r} is not finite")


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: float | LearningRateSchedule = 1e-2,
        *,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(learning_rate)
        self.momentum = _require_beta(momentum, "momentum")
        self.weight_decay = _require_non_negative(weight_decay, "weight_decay")
        self._velocity: dict[str, np.ndarray] = {}

    def step(
        self,
        parameters: MutableMapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        self._validate(parameters, gradients)
        rate = self.schedule(self.step_count)
        for name, parameter in parameters.items():
            gradient = np.asarray(gradients[name]) + self.weight_decay * parameter
            velocity = self._velocity.setdefault(name, np.zeros_like(parameter))
            velocity[...] = self.momentum * velocity + gradient
            parameter[...] -= rate * velocity
        self.step_count += 1


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float | LearningRateSchedule = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = _require_beta(beta1, "beta1")
        self.beta2 = _require_beta(beta2, "beta2")
        self.epsilon = _require_positive(epsilon, "epsilon")
        self.weight_decay = _require_non_negative(weight_decay, "weight_decay")
        self._first: dict[str, np.ndarray] = {}
        self._second: dict[str, np.ndarray] = {}

    def step(
        self,
        parameters: MutableMapping[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
    ) -> None:
        self._validate(parameters, gradients)
        rate = self.schedule(self.step_count)
        time = self.step_count + 1
        for name, parameter in parameters.items():
            gradient = np.asarray(gradients[name]) + self.weight_decay * parameter
            first = self._first.setdefault(name, np.zeros_like(parameter))
            second = self._second.setdefault(name, np.zeros_like(parameter))
            first[...] = self.beta1 * first + (1.0 - self.beta1) * gradient
            second[...] = self.beta2 * second + (1.0 - self.beta2) * np.square(gradient)
            corrected_first = first / (1.0 - self.beta1**time)
            corrected_second = second / (1.0 - self.beta2**time)
            parameter[...] -= rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)
        self.step_count += 1
