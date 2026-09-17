"""Transparent trainable molecular baseline used to validate the plugin contract.

This is a synthetic demonstration: ``MolecularPropertyBaseline`` is a linear
model over deterministic molecular descriptors. It is provided to exercise the
scientific-model contract (forward / loss-and-gradients / persistence) and is
*not* a trained or pretrained quantum-chemistry model. The ``formation_energy``
label is illustrative and must not be read as a real scientific prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

from sciai.core import units
from sciai.domains.molecule.data import Molecule
from sciai.exceptions import ValidationError
from sciai.models import ScienceModel


class MolecularPropertyBaseline(ScienceModel):
    """Linear molecular-descriptor baseline, not a pretrained SOTA model."""

    model_type = "molecular-property-baseline"
    domain = "molecule"

    def __init__(
        self,
        *,
        property_name: str = "formation_energy",
        weights: list[float] | None = None,
        bias: float = 0.0,
        output_unit: str = "eV",
    ) -> None:
        initial = [0.0] * 5 if weights is None else list(weights)
        if len(initial) != 5:
            raise ValidationError("Molecular baseline expects exactly five descriptor weights")
        super().__init__(
            property_name=property_name,
            weights=list(initial),
            bias=float(bias),
            output_unit=output_unit,
        )
        self.property_name = property_name
        # Validate and normalize the output unit against the ontology registry.
        self.output_unit = units.parse(output_unit).symbol
        self.weights: np.ndarray = np.asarray(initial, dtype=float)
        self.bias: np.ndarray = np.asarray([bias], dtype=float)
        if not np.all(np.isfinite(self.weights)):
            raise ValidationError("Molecular baseline weights must be finite")
        if not np.all(np.isfinite(self.bias)):
            raise ValidationError("Molecular baseline bias must be finite")

    @staticmethod
    def _features(inputs: Any) -> tuple[np.ndarray, bool]:
        single = isinstance(inputs, (Molecule, str))
        records = [inputs] if single else list(inputs)
        if not records:
            raise ValidationError("Molecular model inputs cannot be empty")
        molecules = [
            Molecule.from_smiles(record) if isinstance(record, str) else record
            for record in records
        ]
        if not all(isinstance(record, Molecule) for record in molecules):
            raise ValidationError("Molecular models accept SMILES strings or Molecule objects")
        return np.stack([molecule.descriptors() for molecule in molecules]), single

    def forward(self, inputs: Any, **kwargs: Any) -> float | np.ndarray:
        requested = kwargs.pop("property", self.property_name)
        if kwargs:
            raise TypeError(f"Unexpected molecular model options: {sorted(kwargs)}")
        if requested != self.property_name:
            raise ValidationError(
                f"This model predicts {self.property_name!r}, not {requested!r}"
            )
        features, single = self._features(inputs)
        # Compute in the weights' dtype so a future ScienceModel.to_precision
        # (e.g. float32) is honored without upcasting.
        feats = np.asarray(features, dtype=self.weights.dtype)
        weights = np.asarray(self.weights, dtype=self.weights.dtype)
        bias = np.asarray(self.bias, dtype=self.weights.dtype)
        predictions = feats @ weights + bias[0]
        return float(predictions[0]) if single else predictions

    def trainable_parameters(self) -> MutableMapping[str, np.ndarray]:
        return {"weights": self.weights, "bias": self.bias}

    def to_precision(self, dtype: Any) -> MolecularPropertyBaseline:
        """Cast the baseline's mutable NumPy parameters in a single operation."""
        target = np.dtype(dtype)
        if target.kind not in {"f", "c"}:
            raise ValidationError("Molecular model precision must be floating or complex")
        if self.weights.dtype != target:
            self.weights = self.weights.astype(target)
            self.bias = self.bias.astype(target)
        return self

    def loss_and_gradients(self, batch: Any) -> tuple[float, Mapping[str, np.ndarray]]:
        records = list(batch)
        if not records:
            raise ValidationError("Molecular training batch cannot be empty")
        molecules, targets = zip(*records, strict=True)
        features, _ = self._features(molecules)
        # Force a 1-D target array and reject shapes that would broadcast.
        target = np.asarray(targets, dtype=self.weights.dtype)
        if target.ndim != 1 or target.shape[0] != features.shape[0]:
            raise ValidationError(
                "Each training record needs exactly one scalar target; "
                f"expected shape ({features.shape[0]},), got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValidationError("Molecular training targets must be finite")

        feats = np.asarray(features, dtype=self.weights.dtype)
        weights = np.asarray(self.weights, dtype=self.weights.dtype)
        bias = np.asarray(self.bias, dtype=self.weights.dtype)
        prediction = feats @ weights + bias[0]
        residual = prediction - target
        loss = float(np.mean(np.square(residual)))
        scale = 2.0 / len(records)
        gradients = {
            "weights": (scale * feats.T @ residual).astype(self.weights.dtype),
            "bias": np.asarray([scale * np.sum(residual)], dtype=self.weights.dtype),
        }
        return loss, gradients

    def state_dict(self) -> Mapping[str, np.ndarray]:
        return {"weights": self.weights, "bias": self.bias}
