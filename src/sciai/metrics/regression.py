"""General regression metrics with streaming O(1) accumulation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sciai.exceptions import ValidationError
from sciai.metrics.base import Metric


class _PairMetric(Metric, ABC):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count: int = 0
        self._sum_abs: float = 0.0
        self._sum_sq: float = 0.0
        self._sum_sq_target: float = 0.0
        self._target_count: int = 0
        self._target_mean: Any = 0.0
        self._target_m2: float = 0.0

    def update(self, prediction: Any, target: Any) -> None:
        prediction_array = np.asarray(prediction)
        target_array = np.asarray(target)
        if prediction_array.shape != target_array.shape:
            raise ValidationError(
                f"Prediction and target shapes differ: "
                f"{prediction_array.shape} != {target_array.shape}"
            )
        if prediction_array.size == 0:
            raise ValidationError(f"{type(self).__name__} received an empty observation")
        if not (np.all(np.isfinite(prediction_array)) and np.all(np.isfinite(target_array))):
            raise ValidationError(f"{type(self).__name__} received non-finite values")
        difference = prediction_array - target_array
        abs_difference = np.abs(difference)
        self._count += int(prediction_array.size)
        self._sum_abs += float(np.sum(abs_difference))
        self._sum_sq += float(np.sum(abs_difference * abs_difference))
        self._sum_sq_target += float(np.sum(np.abs(target_array) * np.abs(target_array)))
        self._accumulate_target(target_array)

    def _accumulate_target(self, target_array: np.ndarray) -> None:
        batch = target_array.ravel()
        batch_size = int(batch.size)
        if batch_size == 0:
            return
        batch_mean = np.sum(batch) / batch_size
        batch_m2 = float(np.sum(np.abs(batch - batch_mean) * np.abs(batch - batch_mean)))
        count = self._target_count + batch_size
        if self._target_count == 0:
            self._target_mean = batch_mean
            self._target_m2 = batch_m2
        else:
            mean_delta = batch_mean - self._target_mean
            self._target_m2 = (
                self._target_m2
                + batch_m2
                + (self._target_count * batch_size / count) * float(np.abs(mean_delta) ** 2)
            )
            self._target_mean = (
                self._target_count * self._target_mean + batch_size * batch_mean
            ) / count
        self._target_count = count

    def _require_observations(self) -> None:
        if self._count == 0:
            raise ValidationError(f"{type(self).__name__} has no observations")

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError


class MeanAbsoluteError(_PairMetric):
    def compute(self) -> float:
        self._require_observations()
        return self._sum_abs / self._count


class RootMeanSquaredError(_PairMetric):
    def compute(self) -> float:
        self._require_observations()
        return float(np.sqrt(self._sum_sq / self._count))


class R2Score(_PairMetric):
    def compute(self) -> float:
        self._require_observations()
        residual = self._sum_sq
        total = self._target_m2
        if total == 0.0:
            return 1.0 if residual == 0.0 else 0.0
        return 1.0 - residual / total


class RelativeError(_PairMetric):
    def __init__(self, *, epsilon: float = 1e-12) -> None:
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValidationError("epsilon must be a positive finite float")
        self.epsilon = epsilon
        super().__init__()

    def compute(self) -> float:
        self._require_observations()
        numerator = float(np.sqrt(self._sum_sq))
        denominator = max(float(np.sqrt(self._sum_sq_target)), self.epsilon)
        return numerator / denominator


MAE = MeanAbsoluteError
RMSE = RootMeanSquaredError
R2 = R2Score
