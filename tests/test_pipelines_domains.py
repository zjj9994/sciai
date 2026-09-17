import numpy as np
import pytest

from sciai import AutoModel, pipeline
from sciai.domains import DOMAIN_CATALOG, load_builtin_plugins
from sciai.domains.fluid import FlowField
from sciai.domains.molecule import Molecule
from sciai.exceptions import ValidationError
from sciai.registry import domain_registry


def test_molecular_pipeline() -> None:
    result = pipeline(
        "property-prediction",
        model="sciai/molecule-baseline",
    )("CCO")
    assert result["property"] == "formation_energy"
    assert result["unit"] == "eV"
    assert isinstance(result["value"], float)


def test_fluid_conservation_pipeline() -> None:
    axis = np.linspace(0.0, 1.0, 9)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    flow = FlowField.from_arrays(
        np.stack([x, -y], axis=-1),
        coordinates=(axis, axis),
    )
    result = pipeline(
        "conservation-check",
        model=AutoModel.from_pretrained("incompressibility-residual"),
    )(flow)
    assert result["value"] < 1e-10


def test_molecule_parser_is_explicit_about_scope() -> None:
    assert Molecule.from_smiles("CC(=O)O").symbols == ("C", "C", "O", "O")
    with pytest.raises(ValidationError):
        Molecule.from_smiles("[NH4+]")


def test_catalog_and_builtin_plugins() -> None:
    load_builtin_plugins()
    assert len(DOMAIN_CATALOG) == 7
    assert len({sub for family in DOMAIN_CATALOG for sub in family.subdomains}) >= 30
    assert domain_registry.contains("molecule")
    assert domain_registry.contains("fluid")
