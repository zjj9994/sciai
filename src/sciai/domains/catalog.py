"""The cross-disciplinary capability map; availability is explicit, never implied."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    PLUGIN_READY = "plugin-ready"
    ROADMAP = "roadmap"


@dataclass(frozen=True)
class DomainFamily:
    key: str
    name: str
    subdomains: tuple[str, ...]
    core_data: tuple[str, ...]
    model_families: tuple[str, ...]
    simulators: tuple[str, ...]
    status: CapabilityStatus = CapabilityStatus.PLUGIN_READY


DOMAIN_CATALOG: tuple[DomainFamily, ...] = (
    DomainFamily(
        "math-physics",
        "Mathematical and Physical Sciences",
        (
            "mathematics",
            "computational-mathematics",
            "statistics",
            "condensed-matter",
            "particle-physics",
            "quantum-mechanics",
            "optics",
            "thermodynamics",
        ),
        ("tensor-field", "wavefunction", "lattice", "cross-section", "spectrum"),
        ("PINN", "quantum-neural-network", "lattice-gauge-model", "wavefunction-fit"),
        ("FEniCS", "COMSOL", "Mathematica", "LAMMPS"),
    ),
    DomainFamily(
        "chemistry-molecule",
        "Chemistry and Molecular Sciences",
        (
            "quantum-chemistry",
            "molecular-dynamics",
            "organic-chemistry",
            "inorganic-chemistry",
            "catalysis",
            "polymer-chemistry",
        ),
        (
            "molecular-graph",
            "electronic-structure",
            "potential-energy-surface",
            "trajectory",
            "spectrum",
        ),
        ("GNN", "DimeNet", "SchNet", "NequIP", "MACE", "molecular-generation"),
        ("VASP", "Gaussian", "LAMMPS", "GROMACS", "ORCA", "CP2K"),
        CapabilityStatus.AVAILABLE,
    ),
    DomainFamily(
        "materials",
        "Materials Science and Engineering",
        (
            "metals",
            "ceramics",
            "polymers",
            "composites",
            "semiconductors",
            "functional-materials",
        ),
        ("crystal", "defect", "microstructure", "stress-strain-field", "property-curve"),
        ("crystal-property", "alloy-design", "defect-evolution", "phase-transition", "multiscale"),
        ("VASP", "Materials Studio", "ABAQUS", "ANSYS", "Phase-Field"),
    ),
    DomainFamily(
        "life-medicine",
        "Life Sciences and Medicine",
        (
            "molecular-biology",
            "cell-biology",
            "physiology",
            "pharmacology",
            "neuroscience",
            "medical-imaging",
        ),
        (
            "protein-structure",
            "sequence",
            "cell-image",
            "electrophysiology",
            "medical-image",
            "drug-molecule",
        ),
        ("protein-structure", "affinity", "segmentation", "gene-expression", "eeg-analysis"),
        ("AutoDock", "NAMD", "FSL", "3D Slicer"),
    ),
    DomainFamily(
        "earth-environment",
        "Earth and Environmental Sciences",
        ("climate", "atmosphere", "ocean", "geology", "hydrology", "ecology", "seismology"),
        (
            "grid-field",
            "remote-sensing",
            "seismic-waveform",
            "hydrology-series",
            "geological-section",
            "dispersion-field",
        ),
        (
            "climate-forecast",
            "extreme-weather",
            "ocean-circulation",
            "seismic-wave",
            "pollutant-dispersion",
        ),
        ("WRF", "CESM", "FVCOM", "OpenFOAM", "MODIS"),
    ),
    DomainFamily(
        "engineering",
        "Engineering Sciences and Technology",
        (
            "fluid-mechanics",
            "solid-mechanics",
            "thermal-engineering",
            "electrical-engineering",
            "mechanical-engineering",
            "civil-engineering",
            "aerospace",
        ),
        (
            "flow-field",
            "stress-field",
            "temperature-field",
            "electromagnetic-field",
            "mesh",
            "operating-condition",
        ),
        (
            "flow-surrogate",
            "structural-prediction",
            "heat-optimization",
            "electromagnetic-surrogate",
            "fatigue-life",
        ),
        ("OpenFOAM", "Fluent", "ABAQUS", "ANSYS", "COMSOL", "MATLAB/Simulink"),
        CapabilityStatus.AVAILABLE,
    ),
    DomainFamily(
        "astronomy-space",
        "Astronomy and Space Sciences",
        (
            "astrophysics",
            "cosmology",
            "planetary-science",
            "space-physics",
            "radio-astronomy",
        ),
        (
            "spectrum",
            "astronomical-image",
            "gravitational-wave",
            "cosmology-simulation",
            "orbit",
        ),
        (
            "galaxy-formation",
            "gravitational-wave-detection",
            "exoplanet",
            "large-scale-structure",
        ),
        ("GADGET", "AREPO", "CASA", "Kepler"),
    ),
)


def get_domain_family(key: str) -> DomainFamily:
    normalized = key.strip().lower().replace("_", "-")
    for family in DOMAIN_CATALOG:
        if family.key == normalized:
            return family
    raise KeyError(f"Unknown domain family {key!r}")
