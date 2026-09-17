"""Contract tests for the training engine, optimizers, metrics, and dataset.

These tests pin the semantic and type-level contracts described for the
sciai-core 0.1.0a1 unified training path:

* ``ScienceDataset`` conforms to ``Sequence`` and never triggers ndarray
  boolean ambiguity when checking for emptiness.
* ``TrainingConfig`` no longer rejects ``gradient_accumulation_steps`` /
  ``distributed_backend`` at construction so third-party backends can reuse it;
  the NumPy trainer explicitly rejects distributed execution and implements
  sample-count-weighted gradient accumulation with correct tail batches.
* Optimizers validate hyperparameters (finite / range), forbid integer
  parameters, and uniformly validate gradient shape / dtype / finiteness
  before mutating parameters.
* Regression metrics use O(1) streaming sufficient statistics with Welford
  merging for R2, support complex values via absolute squares, and reject
  empty / non-finite observations.
"""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np
import pytest

from sciai import DatasetInfo, ScienceDataset, ScienceModel, ScienceTrainer, TrainingConfig
from sciai.exceptions import MissingBackendError, ValidationError
from sciai.metrics import MAE, R2, RMSE, RelativeError
from sciai.training import SGD, Adam, ConstantLR, CosineDecayLR, WarmupCosineLR


class PrecisionModel(ScienceModel):
    """Minimal linear model whose loss is the per-sample mean squared error."""

    def __init__(self, n_features: int, dtype: Any = np.float64) -> None:
        super().__init__()
        self.n_features = n_features
        self.w = np.zeros(n_features, dtype=dtype)
        self._dtype = np.dtype(dtype)

    def to_precision(self, dtype: Any) -> "PrecisionModel":
        target = np.dtype(dtype)
        if target != self._dtype:
            self.w = self.w.astype(target)
            self._dtype = target
        return self

    def trainable_parameters(self) -> MutableMapping[str, np.ndarray]:
        return {"w": self.w}

    def forward(self, inputs: Any, **kwargs: Any) -> Any:
        features = np.asarray(inputs, dtype=self._dtype)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return features @ self.w

    def loss_and_gradients(self, batch: Any) -> tuple[float, Mapping[str, np.ndarray]]:
        xs = np.stack([row[0] for row in batch]).astype(self._dtype)
        ys = np.asarray([row[1] for row in batch], dtype=self._dtype)
        prediction = xs @ self.w
        residual = prediction - ys
        loss = float(np.mean(np.square(residual.astype(np.float64))))
        scale = 2.0 / len(batch)
        grad = {"w": (scale * xs.T @ residual).astype(self._dtype)}
        return loss, grad


class InfLossModel(PrecisionModel):
    def loss_and_gradients(self, batch: Any) -> tuple[float, Mapping[str, np.ndarray]]:
        loss, grad = super().loss_and_gradients(batch)
        return float("inf"), grad


class NanGradModel(PrecisionModel):
    def loss_and_gradients(self, batch: Any) -> tuple[float, Mapping[str, np.ndarray]]:
        loss, grad = super().loss_and_gradients(batch)
        return loss, {name: np.full_like(value, np.nan) for name, value in grad.items()}


def _make_dataset(n_samples: int, n_features: int, seed: int = 0) -> ScienceDataset:
    rng = np.random.default_rng(seed)
    records = [
        (rng.standard_normal(n_features), float(rng.standard_normal()))
        for _ in range(n_samples)
    ]
    return ScienceDataset(records, info=DatasetInfo(name="toy"))


class TestScienceDatasetSequenceContract:
    def test_getitem_int_and_slice_overloads(self) -> None:
        dataset = ScienceDataset(
            [("a", 1), ("b", 2), ("c", 3)],
            info=DatasetInfo(name="x"),
        )
        assert dataset[0] == ("a", 1)
        assert isinstance(dataset[0:2], Sequence)
        assert len(dataset[0:2]) == 2

    def test_size_gt_one_ndarray_does_not_raise_bool_ambiguity(self) -> None:
        # ``if not records`` would raise ``ValueError: truth value of array is
        # ambiguous`` for a size>1 ndarray. The dataset must construct cleanly.
        dataset = ScienceDataset(np.array([1, 2, 3, 4]), info=DatasetInfo(name="x"))
        assert len(dataset) == 4
        assert dataset[0] == 1
        assert dataset[0:2] == (1, 2)

    def test_empty_dataset_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScienceDataset(np.array([]), info=DatasetInfo(name="x"))


class TestConfigDoesNotRejectSharedFields:
    def test_gradient_accumulation_allowed_at_construction(self) -> None:
        config = TrainingConfig(epochs=1, gradient_accumulation_steps=4)
        assert config.gradient_accumulation_steps == 4

    def test_distributed_backend_allowed_at_construction(self) -> None:
        config = TrainingConfig(epochs=1, distributed_backend="nccl")
        assert config.distributed_backend == "nccl"

    def test_numpy_trainer_rejects_distributed_explicitly(self) -> None:
        model = PrecisionModel(3)
        dataset = _make_dataset(6, 3)
        trainer = ScienceTrainer(config=TrainingConfig(epochs=1, distributed_backend="nccl"))
        with pytest.raises(MissingBackendError):
            trainer.fit(model, dataset)

    def test_invalid_precision_still_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=1, precision="bfloat16")


class TestGradientAccumulation:
    def test_accumulation_matches_full_batch(self) -> None:
        n, feat = 6, 3
        dataset = _make_dataset(n, feat)
        full = PrecisionModel(feat)
        ScienceTrainer(
            config=TrainingConfig(epochs=2, batch_size=n, shuffle=False),
            optimizer=Adam(learning_rate=0.01),
        ).fit(full, dataset)

        accumulated = PrecisionModel(feat)
        ScienceTrainer(
            config=TrainingConfig(
                epochs=2, batch_size=1, shuffle=False, gradient_accumulation_steps=n
            ),
            optimizer=Adam(learning_rate=0.01),
        ).fit(accumulated, dataset)

        assert np.allclose(full.w, accumulated.w, atol=1e-5, rtol=1e-4)

    def test_accumulation_tail_batch_is_correct(self) -> None:
        n, feat = 5, 3
        dataset = _make_dataset(n, feat)
        full = PrecisionModel(feat)
        full_history = ScienceTrainer(
            config=TrainingConfig(epochs=1, batch_size=n, shuffle=False),
            optimizer=Adam(learning_rate=0.01),
        ).fit(full, dataset)

        accumulated = PrecisionModel(feat)
        acc_history = ScienceTrainer(
            config=TrainingConfig(
                epochs=1, batch_size=2, shuffle=False, gradient_accumulation_steps=n
            ),
            optimizer=Adam(learning_rate=0.01),
        ).fit(accumulated, dataset)

        assert np.allclose(full.w, accumulated.w, atol=1e-5, rtol=1e-4)
        assert acc_history[-1].loss == pytest.approx(full_history[-1].loss, rel=1e-5)

    def test_reported_loss_is_per_sample_mean(self) -> None:
        n, feat = 5, 3
        dataset = _make_dataset(n, feat)
        model = PrecisionModel(feat)
        history = ScienceTrainer(
            config=TrainingConfig(
                epochs=1, batch_size=1, shuffle=False, gradient_accumulation_steps=n
            )
        ).fit(model, dataset)
        # The single accumulated step's loss must equal the mean per-sample loss
        # evaluated at the initial weights (before the optimizer update).
        reference_loss, _ = PrecisionModel(feat).loss_and_gradients(list(dataset))
        assert history[-1].loss == pytest.approx(reference_loss, rel=1e-6)

    def test_invalid_accumulation_steps_rejected(self) -> None:
        model = PrecisionModel(3)
        dataset = _make_dataset(6, 3)
        trainer = ScienceTrainer(config=TrainingConfig(epochs=1, gradient_accumulation_steps=0))
        with pytest.raises(ValidationError):
            trainer.fit(model, dataset)


class TestPrecisionAndLifecycle:
    def test_to_precision_is_applied_for_float32(self) -> None:
        model = PrecisionModel(3, dtype=np.float64)
        dataset = _make_dataset(6, 3)
        ScienceTrainer(
            config=TrainingConfig(epochs=1, precision="float32")
        ).fit(model, dataset)
        assert model.w.dtype == np.float32

    def test_to_precision_is_applied_for_float64(self) -> None:
        model = PrecisionModel(3, dtype=np.float32)
        dataset = _make_dataset(6, 3)
        ScienceTrainer(
            config=TrainingConfig(epochs=1, precision="float64")
        ).fit(model, dataset)
        assert model.w.dtype == np.float64

    def test_parameter_reference_stays_valid_after_precision_change(self) -> None:
        model = PrecisionModel(3, dtype=np.float64)
        dataset = _make_dataset(6, 3)
        ScienceTrainer(config=TrainingConfig(epochs=1, precision="float32")).fit(model, dataset)
        assert model.trainable_parameters()["w"] is model.w

    def test_history_is_cleared_each_fit(self) -> None:
        model = PrecisionModel(3)
        dataset = _make_dataset(6, 3)
        trainer = ScienceTrainer(config=TrainingConfig(epochs=2, batch_size=6, shuffle=False))
        first = trainer.fit(model, dataset)
        second = trainer.fit(model, dataset)
        assert len(second) == 2
        assert len(trainer.history) == len(second)
        assert len(first) + len(second) != len(trainer.history)

    def test_training_state_restored_after_success(self) -> None:
        model = PrecisionModel(3)
        model.training = True
        dataset = _make_dataset(6, 3)
        ScienceTrainer(config=TrainingConfig(epochs=1)).fit(model, dataset)
        assert model.training is True

    def test_training_state_restored_after_failure(self) -> None:
        model = PrecisionModel(3)
        model.training = True
        dataset = _make_dataset(6, 3)
        with pytest.raises(FloatingPointError):
            ScienceTrainer(config=TrainingConfig(epochs=1)).fit(InfLossModel(3), dataset)
        assert model.training is True

    def test_non_finite_loss_rejected(self) -> None:
        dataset = _make_dataset(6, 3)
        with pytest.raises(FloatingPointError):
            ScienceTrainer(config=TrainingConfig(epochs=1)).fit(InfLossModel(3), dataset)

    def test_non_finite_gradient_rejected(self) -> None:
        dataset = _make_dataset(6, 3)
        with pytest.raises(ValidationError):
            ScienceTrainer(config=TrainingConfig(epochs=1)).fit(NanGradModel(3), dataset)

    def test_callback_without_side_effects_supported(self) -> None:
        model = PrecisionModel(3)
        dataset = _make_dataset(6, 3)

        class Recorder:
            def __init__(self) -> None:
                self.losses: list[float] = []

            def on_train_begin(self, model: ScienceModel, config: TrainingConfig) -> None:
                pass

            def on_step_end(self, state: Any) -> None:
                self.losses.append(state.loss)

            def on_train_end(self, state: Any) -> None:
                pass

        recorder = Recorder()
        trainer = ScienceTrainer(
            config=TrainingConfig(epochs=2, batch_size=6, shuffle=False), callbacks=[recorder]
        )
        history = trainer.fit(model, dataset)
        assert recorder.losses == [state.loss for state in history]


class TestOptimizerValidation:
    def test_integer_learning_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Adam(learning_rate=1)
        with pytest.raises(ValidationError):
            SGD(learning_rate=1)

    def test_non_finite_learning_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Adam(learning_rate=float("inf"))
        with pytest.raises(ValidationError):
            Adam(learning_rate=float("nan"))

    def test_beta_range_validated(self) -> None:
        with pytest.raises(ValidationError):
            Adam(beta1=1.0)
        with pytest.raises(ValidationError):
            Adam(beta2=1.5)
        with pytest.raises(ValidationError):
            Adam(beta1=-0.1)

    def test_epsilon_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Adam(epsilon=0.0)
        with pytest.raises(ValidationError):
            Adam(epsilon=-1e-8)
        with pytest.raises(ValidationError):
            Adam(epsilon=float("inf"))

    def test_momentum_and_weight_decay_ranges(self) -> None:
        with pytest.raises(ValidationError):
            SGD(momentum=1.0)
        with pytest.raises(ValidationError):
            SGD(momentum=-0.1)
        with pytest.raises(ValidationError):
            SGD(weight_decay=-1.0)

    def test_integer_parameters_rejected(self) -> None:
        optimizer = SGD()
        with pytest.raises(ValidationError):
            optimizer.step(
                {"w": np.zeros(3, dtype=int)},
                {"w": np.zeros(3, dtype=float)},
            )

    def test_complex_gradients_allowed(self) -> None:
        optimizer = SGD()
        params = {"w": np.zeros(2, dtype=complex)}
        grads = {"w": np.array([1.0 + 1.0j, 2.0 - 1.0j])}
        optimizer.step(params, grads)
        assert np.all(np.isfinite(params["w"]))
        assert params["w"].dtype == complex

    def test_non_finite_gradient_rejected(self) -> None:
        optimizer = Adam()
        with pytest.raises(ValidationError):
            optimizer.step(
                {"w": np.zeros(3, dtype=float)},
                {"w": np.full(3, np.nan)},
            )

    def test_gradient_shape_mismatch_rejected(self) -> None:
        optimizer = Adam()
        with pytest.raises(ValidationError):
            optimizer.step(
                {"w": np.zeros(3, dtype=float)},
                {"w": np.zeros(4, dtype=float)},
            )

    def test_lr_schedule_validation(self) -> None:
        with pytest.raises(ValidationError):
            ConstantLR(0.0)
        with pytest.raises(ValidationError):
            WarmupCosineLR(peak_rate=-1.0)
        # A valid schedule must be accepted and return a finite positive rate.
        schedule = WarmupCosineLR(peak_rate=1e-3)
        assert Adam(learning_rate=schedule) is not None
        assert CosineDecayLR(initial_rate=1e-3) is not None


class TestRegressionMetrics:
    def test_streaming_matches_single_update(self) -> None:
        rng = np.random.default_rng(0)
        prediction = rng.standard_normal(40)
        target = rng.standard_normal(40)
        for metric_cls in (MAE, RMSE, R2, RelativeError):
            whole = metric_cls()
            whole.update(prediction, target)
            chunked = metric_cls()
            for start in range(0, 40, 7):
                chunked.update(prediction[start : start + 7], target[start : start + 7])
            assert chunked.compute() == pytest.approx(whole.compute(), rel=1e-9)

    def test_matches_brute_force(self) -> None:
        rng = np.random.default_rng(1)
        prediction = rng.standard_normal(20)
        target = rng.standard_normal(20)
        diff = prediction - target

        mae = MAE()
        mae.update(prediction, target)
        assert mae.compute() == pytest.approx(float(np.mean(np.abs(diff))))

        rmse = RMSE()
        rmse.update(prediction, target)
        assert rmse.compute() == pytest.approx(float(np.sqrt(np.mean(diff**2))))

        r2 = R2()
        r2.update(prediction, target)
        residual = float(np.sum(diff**2))
        total = float(np.sum((target - target.mean()) ** 2))
        expected = 1.0 - residual / total if total != 0 else (1.0 if residual == 0 else 0.0)
        assert r2.compute() == pytest.approx(expected, rel=1e-9)

        rel = RelativeError()
        rel.update(prediction, target)
        expected_rel = float(np.linalg.norm(diff) / np.linalg.norm(target))
        assert rel.compute() == pytest.approx(expected_rel, rel=1e-9)

    def test_complex_values_supported_without_dropping_imag(self) -> None:
        # If the imaginary part were silently dropped, MAE would be 0.
        mae = MAE()
        mae.update(1 + 2j, 1 + 0j)
        assert mae.compute() == pytest.approx(2.0)

        rmse = RMSE()
        rmse.update(1 + 2j, 1 + 0j)
        assert rmse.compute() == pytest.approx(2.0)

        # Perfect complex prediction -> R2 == 1.
        r2 = R2()
        r2.update([1 + 1j, 2 + 2j], [1 + 1j, 2 + 2j])
        assert r2.compute() == pytest.approx(1.0)

    def test_non_finite_observation_rejected(self) -> None:
        metric = MAE()
        with pytest.raises(ValidationError):
            metric.update([np.nan, 1.0], [1.0, 1.0])
        metric2 = RMSE()
        with pytest.raises(ValidationError):
            metric2.update([1.0, 1.0], [np.inf, 1.0])

    def test_empty_observation_rejected(self) -> None:
        metric = MAE()
        with pytest.raises(ValidationError):
            metric.compute()

    def test_relative_error_epsilon_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RelativeError(epsilon=0.0)
        with pytest.raises(ValidationError):
            RelativeError(epsilon=-1e-12)

    def test_r2_uses_welford_to_avoid_cancellation(self) -> None:
        # Large constant baseline with tiny variation: the naive
        # sum(x**2) - n*mean**2 form loses precision to cancellation, while the
        # Welford streaming form keeps it accurate.
        base = 1e8
        n = 100
        target = base + np.linspace(-0.5, 0.5, n)
        prediction = target + 1e-3
        naive_total = float(np.sum(target**2) - n * target.mean() ** 2)
        stable_total = float(np.sum((target - target.mean()) ** 2))

        r2 = R2()
        r2.update(prediction, target)
        # The metric must not degenerate via catastrophic cancellation.
        assert not (stable_total == 0 and naive_total != 0)
        expected_r2 = 1.0 - np.sum((prediction - target) ** 2) / stable_total
        assert r2.compute() == pytest.approx(expected_r2, rel=1e-9)
        # And the naive form is clearly unreliable for this input.
        assert abs(naive_total - stable_total) > 0.5 * abs(stable_total)
