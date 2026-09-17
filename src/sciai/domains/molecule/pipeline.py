"""Molecular task pipelines.

The ``property-prediction`` pipeline is a thin, contract-focused wrapper around
:class:`MolecularPropertyBaseline`. It is a synthetic demonstration of the
pipeline API and does not constitute a trained scientific inference service.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sciai.domains.molecule.data import Molecule
from sciai.domains.molecule.model import MolecularPropertyBaseline
from sciai.exceptions import ValidationError
from sciai.pipelines import SciencePipeline


class MolecularPropertyPipeline(SciencePipeline):
    """Property-prediction pipeline bound to a MolecularPropertyBaseline."""

    task = "property-prediction"

    def __init__(self, model: MolecularPropertyBaseline) -> None:
        # Explicit subtype with a runtime type check so a mismatched model fails
        # loudly instead of producing ambiguous downstream errors.
        if not isinstance(model, MolecularPropertyBaseline):
            raise ValidationError(
                f"MolecularPropertyPipeline requires a MolecularPropertyBaseline, "
                f"got {type(model).__name__}"
            )
        super().__init__(model)
        self.model: MolecularPropertyBaseline = model

    def preprocess(self, inputs: Any, **kwargs: Any) -> Any:
        del kwargs
        if isinstance(inputs, str):
            return Molecule.from_smiles(inputs)
        if isinstance(inputs, Molecule):
            return inputs
        # Batch inputs (a list of SMILES strings or Molecule objects) are passed
        # through to the model, which handles batched descriptors.
        return inputs

    def postprocess(self, outputs: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        array = np.asarray(outputs, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValidationError("Molecular pipeline produced non-finite outputs")
        base = {
            "property": self.model.property_name,
            "unit": self.model.output_unit,
            "model": self.model.model_type,
        }
        # Explicitly handle scalar and batched outputs rather than relying on
        # implicit broadcasting/float coercion of an arbitrary array.
        if array.ndim == 0 or array.size == 1:
            return {**base, "value": float(array.ravel()[0])}
        return {
            **base,
            "values": [float(value) for value in array.ravel()],
            "count": int(array.size),
        }
