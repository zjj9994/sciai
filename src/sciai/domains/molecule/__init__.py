"""Built-in molecular-science reference plugin."""

from __future__ import annotations

from sciai.auto import DatasetProvider, ModelProvider, PipelineProvider
from sciai.data import DatasetInfo, ScienceDataset
from sciai.domains.base import DomainPlugin
from sciai.domains.molecule.data import Molecule
from sciai.domains.molecule.model import MolecularPropertyBaseline
from sciai.domains.molecule.pipeline import MolecularPropertyPipeline


def _toy_dataset() -> ScienceDataset[tuple[Molecule, float]]:
    records = [
        (Molecule.from_smiles("C"), -0.5),
        (Molecule.from_smiles("CC"), -1.0),
        (Molecule.from_smiles("CO"), -0.8),
        (Molecule.from_smiles("CCO"), -1.2),
        (Molecule.from_smiles("O"), -0.2),
    ]
    return ScienceDataset(
        records,
        info=DatasetInfo(
            name="toy-molecules",
            domain="molecule",
            version="1",
            license="CC0-1.0",
            description="Synthetic records for API tests; not a scientific benchmark.",
        ),
    )


plugin = DomainPlugin(
    name="molecule",
    version="0.1.0",
    description="Molecular graphs and a transparent trainable property baseline.",
    subdomains=(
        "quantum-chemistry",
        "molecular-dynamics",
        "organic-chemistry",
        "inorganic-chemistry",
        "catalysis",
        "polymer-chemistry",
    ),
    data_types=(Molecule,),
    models={
        "molecule-baseline": ModelProvider(
            factory=MolecularPropertyBaseline,
            domain="molecule",
            tasks=("property-prediction",),
            description="Trainable descriptor baseline; no pretrained scientific claims.",
            pretrained=False,
        ),
        "molecular-property-baseline": ModelProvider(
            factory=MolecularPropertyBaseline,
            domain="molecule",
            tasks=("property-prediction",),
            description="Artifact loader for the molecular baseline model type.",
        ),
    },
    datasets={
        "toy-molecules": DatasetProvider(
            factory=_toy_dataset,
            domain="molecule",
            description="Synthetic test-only molecular records.",
        )
    },
    pipelines={
        "molecule/property-prediction": PipelineProvider(
            factory=MolecularPropertyPipeline,
            domains=("molecule",),
        )
    },
    metadata={
        "production_integrations": ("MACE", "NequIP", "SchNet", "DimeNet", "RDKit"),
        "integration_status": "plugin-ready",
    },
)

GDModel = MolecularPropertyBaseline

__all__ = [
    "GDModel",
    "MolecularPropertyBaseline",
    "MolecularPropertyPipeline",
    "Molecule",
    "plugin",
]
