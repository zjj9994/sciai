"""Contract tests for the conservative SMILES parser and molecular baseline.

These tests pin the scientific boundaries of the molecule domain:
the strict SMILES subset (atom/branch/bond/ring-closure rules and implied
aromatic bond orders), the model's input/parameter/label validation, and the
numeric correctness of its analytic gradients via finite differences.
"""

import numpy as np
import pytest

from sciai.domains.molecule.data import Molecule
from sciai.domains.molecule.model import MolecularPropertyBaseline
from sciai.domains.molecule.pipeline import MolecularPropertyPipeline
from sciai.exceptions import ValidationError
from sciai.models import ScienceModel


# --------------------------------------------------------------------------- #
# SMILES parser: accepted molecules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles, n_atoms",
    [
        ("C", 1),
        ("CC", 2),
        ("CCO", 3),
        ("CC(=O)O", 4),  # acetic acid: symbols (C, C, O, O)
        ("C1CCCCC1", 6),  # cyclohexane
        ("c1ccccc1", 6),  # benzene
        ("Cc1ccccc1", 7),  # toluene
        ("n1ccccc1", 6),  # pyridine
        ("C=CC", 3),  # double bond
        ("C#N", 2),  # triple bond
        ("C(C)(C)C", 4),  # multiple branches
    ],
)
def test_parser_accepts_valid_subset(smiles: str, n_atoms: int) -> None:
    mol = Molecule.from_smiles(smiles)
    assert len(mol.symbols) == n_atoms
    # Every undirected edge appears as two directed edges.
    assert mol.bond_orders.shape[0] == mol.edge_index.shape[1]
    assert mol.edge_index.shape[1] == 2 * (mol.edge_index.shape[1] // 2)


def test_parser_preserves_symbols_for_acetic_acid() -> None:
    assert Molecule.from_smiles("CC(=O)O").symbols == ("C", "C", "O", "O")


# --------------------------------------------------------------------------- #
# SMILES parser: rejected molecules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "smiles",
    [
        "",  # empty
        "C=",  # trailing bond
        "=C",  # leading bond
        "C()",  # empty branch
        "C11",  # ring closure self-loop (digit twice on one atom)
        "C1C1",  # duplicate bond between the same two atoms
        "C(=O",  # unbalanced branch
        ")C",  # branch close before open
        "C1CC",  # unclosed ring closure
        "C(=)",  # trailing bond inside a branch
        "CC(C",  # unbalanced branch at end
        "C=1CCCCC-1",  # conflicting ring-closure bond orders
        "[NH4+]",  # bracket atom unsupported
        "C%10",  # multi-digit ring closure unsupported
        "C.C",  # disconnected components unsupported
    ],
)
def test_parser_rejects_invalid_smiles(smiles: str) -> None:
    with pytest.raises(ValidationError):
        Molecule.from_smiles(smiles)


# --------------------------------------------------------------------------- #
# Aromatic bond orders and ring-closure orders
# --------------------------------------------------------------------------- #
def test_benzene_is_all_aromatic_1_5() -> None:
    mol = Molecule.from_smiles("c1ccccc1")
    assert mol.symbols == ("C", "C", "C", "C", "C", "C")
    assert set(np.unique(mol.bond_orders)) == {1.5}


def test_toluene_mixed_bond_is_single_not_aromatic() -> None:
    # Atom 0 is aliphatic C, atom 1 is aromatic c: the connecting bond must be
    # a single bond, never the implied aromatic 1.5. The ring closure stays 1.5.
    mol = Molecule.from_smiles("Cc1ccccc1")
    # The mixed aliphatic/aromatic bond (atom 0 -> atom 1) is single, and its
    # directed reverse is also single; every aromatic ring bond is 1.5.
    assert mol.bond_orders[0] == pytest.approx(1.0)
    assert mol.bond_orders[1] == pytest.approx(1.0)
    assert set(np.unique(mol.bond_orders)) == {1.0, 1.5}


def test_explicit_bond_wins_over_implied_aromatic() -> None:
    # c-c is an explicit single bond between two aromatic atoms -> 1.0.
    mol = Molecule.from_smiles("c-c")
    assert mol.bond_orders[0] == pytest.approx(1.0)


def test_ring_closure_explicit_double_both_positions() -> None:
    # Explicit closure bond at either end yields a 2.0 ring bond, not 1.5.
    for smiles in ("c=1ccccc1", "c1=ccccc1"):
        mol = Molecule.from_smiles(smiles)
        assert 2.0 in set(np.unique(mol.bond_orders))


# --------------------------------------------------------------------------- #
# Model parameter and input validation
# --------------------------------------------------------------------------- #
def test_model_rejects_empty_weights_list() -> None:
    # weights=[] must be rejected, not silently treated as five zeros.
    with pytest.raises(ValidationError):
        MolecularPropertyBaseline(weights=[])


def test_model_rejects_wrong_weight_count() -> None:
    with pytest.raises(ValidationError):
        MolecularPropertyBaseline(weights=[1.0, 2.0, 3.0])


def test_model_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValidationError):
        MolecularPropertyBaseline(weights=[1.0, 2.0, float("nan"), 4.0, 5.0])
    with pytest.raises(ValidationError):
        MolecularPropertyBaseline(weights=[1.0] * 5, bias=float("inf"))


def test_model_rejects_unknown_output_unit() -> None:
    with pytest.raises(ValidationError):
        MolecularPropertyBaseline(output_unit="wibble")


def test_model_rejects_empty_features() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    with pytest.raises(ValidationError):
        model._features([])


def test_model_preserves_formation_energy_label() -> None:
    model = MolecularPropertyBaseline(weights=[0.5] * 5)
    assert model.property_name == "formation_energy"
    assert model.output_unit == "eV"


# --------------------------------------------------------------------------- #
# Forward: scalar and batch
# --------------------------------------------------------------------------- #
def test_forward_scalar_returns_float() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    out = model("CCO")
    assert isinstance(out, float)


def test_forward_batch_returns_array() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    out = model(["CCO", "C", "CC"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


# --------------------------------------------------------------------------- #
# loss_and_gradients: label shape and broadcast guards
# --------------------------------------------------------------------------- #
def test_loss_rejects_empty_batch() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    with pytest.raises(ValidationError):
        model.loss_and_gradients([])


def test_loss_rejects_wrong_target_shape() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    # A 2-D target would broadcast against the (n, 5) features instead of
    # raising; the explicit shape check must reject it.
    batch = [
        (Molecule.from_smiles("C"), [1.0, 2.0]),
        (Molecule.from_smiles("CC"), [3.0, 4.0]),
    ]
    with pytest.raises(ValidationError):
        model.loss_and_gradients(batch)


def test_loss_rejects_nonfinite_target() -> None:
    model = MolecularPropertyBaseline(weights=[1.0] * 5)
    batch = [(Molecule.from_smiles("C"), float("nan"))]
    with pytest.raises(ValidationError):
        model.loss_and_gradients(batch)


# --------------------------------------------------------------------------- #
# Finite-difference gradient verification
# --------------------------------------------------------------------------- #
def test_gradients_match_finite_difference() -> None:
    weights = np.array([0.3, -0.7, 1.2, 0.5, -0.2])
    model = MolecularPropertyBaseline(weights=list(weights))
    smiles = ["C", "CC", "CCO", "CO", "C1CCCCC1"]
    targets = [0.1, -0.4, 0.8, -0.2, 0.3]
    batch = list(zip(smiles, targets, strict=True))

    _, grads = model.loss_and_gradients(batch)
    analytic_w = np.asarray(grads["weights"])
    analytic_b = float(grads["bias"][0])

    eps = 1e-6
    numeric_w = np.zeros(5)
    for i in range(5):
        wp = weights.copy()
        wp[i] += eps
        wm = weights.copy()
        wm[i] -= eps
        model.weights = wp
        loss_plus = model.loss_and_gradients(batch)[0]
        model.weights = wm
        loss_minus = model.loss_and_gradients(batch)[0]
        model.weights = weights.copy()
        numeric_w[i] = (loss_plus - loss_minus) / (2 * eps)
    assert np.allclose(analytic_w, numeric_w, atol=1e-4)

    # Bias gradient via finite difference.
    model.weights = weights.copy()
    bias0 = float(model.bias[0])
    model.bias = np.asarray([bias0 + eps])
    loss_plus = model.loss_and_gradients(batch)[0]
    model.bias = np.asarray([bias0 - eps])
    loss_minus = model.loss_and_gradients(batch)[0]
    numeric_b = (loss_plus - loss_minus) / (2 * eps)
    assert analytic_b == pytest.approx(numeric_b, abs=1e-4)


# --------------------------------------------------------------------------- #
# dtype awareness (compatibility with ScienceModel.to_precision)
# --------------------------------------------------------------------------- #
def test_gradients_and_forward_respect_weights_dtype() -> None:
    # Simulate ScienceModel.to_precision(float32) by casting the arrays; the
    # computations must preserve that dtype instead of upcasting to float64.
    model = MolecularPropertyBaseline(weights=[0.1, 0.2, 0.3, 0.4, 0.5])
    model.weights = model.weights.astype(np.float32)
    model.bias = model.bias.astype(np.float32)

    batch = [("C", 0.0), ("CC", 1.0)]
    _, grads = model.loss_and_gradients(batch)
    assert grads["weights"].dtype == np.float32
    assert grads["bias"].dtype == np.float32

    batch_out = model.forward(["C", "CC"])
    assert batch_out.dtype == np.float32


def test_to_precision_casts_molecular_parameter_storage() -> None:
    model = MolecularPropertyBaseline(weights=[0.1, 0.2, 0.3, 0.4, 0.5])
    returned = model.to_precision(np.float32)
    assert returned is model
    assert model.weights.dtype == np.float32
    assert model.bias.dtype == np.float32


# --------------------------------------------------------------------------- #
# Pipeline runtime subtype check and scalar/batch outputs
# --------------------------------------------------------------------------- #
class _FakeModel(ScienceModel):
    def forward(self, inputs: object, **kwargs: object) -> object:
        return 0.0


def test_pipeline_rejects_wrong_model_type() -> None:
    with pytest.raises(ValidationError):
        MolecularPropertyPipeline(_FakeModel())


def test_pipeline_scalar_output() -> None:
    model = MolecularPropertyBaseline(weights=[0.0] * 5)
    pipeline = MolecularPropertyPipeline(model)
    result = pipeline("CCO")
    assert isinstance(result["value"], float)
    assert result["property"] == "formation_energy"
    assert result["unit"] == "eV"


def test_pipeline_batch_output() -> None:
    model = MolecularPropertyBaseline(weights=[0.0] * 5)
    pipeline = MolecularPropertyPipeline(model)
    result = pipeline(["CCO", "C", "CC"])
    assert "values" in result
    assert len(result["values"]) == 3
    assert result["count"] == 3
