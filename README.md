# sciai

English | [简体中文](README_ZH-CN.md)

`sciai` is an early, framework-neutral foundation for a unified AI for Science developer experience. Its goal is to make scientific data semantics, models, training, task pipelines, and simulator coupling composable across disciplines without forcing every domain into one machine-learning backend.

> **Status: `0.1.0a1` foundation release.** The stable contracts and NumPy reference path are implemented. Production model weights, benchmark datasets, and native commercial-solver integrations are not bundled yet. Capability status is reported explicitly instead of presenting roadmap items as completed features.

## Why this architecture

- **One scientific ontology:** physical dimensions, units, coordinates, and reference frames are shared across domains.
- **Eight common data structures:** scalar fields, vector fields, tensor fields, graphs, particles, meshes, time series, and spectra.
- **One model contract:** `ScienceModel` standardizes `forward`, `predict`, `train`, `save`, and `load` while leaving tensor execution to backend plugins.
- **Domain plugins, not a monolith:** molecular science and fluid mechanics are built-in reference plugins; third-party packages contribute through `sciai.domains` entry points.
- **Explicit coupling:** simulator adapters exchange named scientific variables through a directed coupling graph.
- **Truthful capability discovery:** the seven domain families and 30+ subdomains are cataloged as `available`, `plugin-ready`, or `roadmap`.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
pytest
```

The repository is named `sciai` and the import package is `sciai`. The PyPI distribution is temporarily named `sciai-core` because the `sciai` distribution name is already occupied by the MindSpore SciAI project.

## Unified API

```python
from sciai import AutoDataset, AutoModel, ScienceTrainer, TrainingConfig, pipeline
from sciai.training import Adam

model = AutoModel.from_pretrained("sciai/molecule-baseline")
dataset = AutoDataset.load("sciai/toy-molecules")

trainer = ScienceTrainer(
    config=TrainingConfig(epochs=100, batch_size=len(dataset)),
    optimizer=Adam(learning_rate=0.01),
)
trainer.fit(model, dataset)

predictor = pipeline("property-prediction", model=model)
print(predictor("CCO"))
```

The built-in molecular model is a transparent trainable descriptor baseline used to prove the model, trainer, artifact, Auto, and pipeline contracts. It is not presented as MACE, SchNet, or another pretrained SOTA model.

## Scientific data

```python
import numpy as np
from sciai import Quantity, ScalarField
from sciai.operators import gradient

length = Quantity(2.5, "km").to("m")
x = np.linspace(0.0, 1.0, 101)
field = ScalarField(values=x**2, coordinates=(x,), unit="K")
derivative = gradient(field)
```

## Domain APIs

```python
from sciai.molecule import Molecule
from sciai.fluid import FlowField, IncompressibilityModel

molecule = Molecule.from_smiles("CCO")
flow = FlowField.from_arrays(velocity, pressure=pressure, coordinates=(x, y))
residual = IncompressibilityModel()(flow)
```

The dependency-free SMILES parser intentionally supports a conservative subset. Full stereochemistry, charged atoms, and canonical chemistry semantics belong in an RDKit-backed plugin.

## Multiphysics coupling

```python
from sciai.multiphysics import CoupledSimulator, InMemoryAdapter

fluid = InMemoryAdapter("fluid", "fluid", {"pressure": 101325.0})
solid = InMemoryAdapter("solid", "solid", {"displacement": 0.0})

simulation = CoupledSimulator([fluid, solid])
simulation.couple("fluid", "solid", variables=["pressure"], interface="wall")
result = simulation.run(steps=10)
```

OpenFOAM, ABAQUS, COMSOL, VASP, GROMACS, and other external programs plug into the same `SimulationAdapter` contract. They are integration targets, not dependencies of the core package.

## Community domain package

```bash
sciai scaffold-domain plasma-physics --destination ./plugins
```

The generated package declares a `sciai.domains` entry point and can independently own domain data, operators, models, datasets, pipelines, and simulator adapters.

## Commands

```bash
sciai catalog
sciai catalog --json
sciai doctor
sciai scaffold-domain <name>
```

## Architecture

```text
Application layer
  AutoModel / AutoDataset / AutoTrainer
  task pipelines / workflow DAG / multiphysics coupling
                         |
Domain plugin layer     | Python entry points + typed registries
  molecule / fluid / community packages
                         |
Universal foundation
  ontology / 8 data types / operators / ScienceModel
  ScienceTrainer / metrics / artifact format / simulation protocol
```

See [docs/architecture.md](docs/architecture.md), [docs/capability-matrix.md](docs/capability-matrix.md), and [docs/roadmap.md](docs/roadmap.md) for contracts and delivery phases.

## Project principles

1. Scientific semantics are validated at boundaries.
2. Core interfaces remain independent of PyTorch, JAX, MindSpore, or TensorFlow.
3. Artifacts use non-executable JSON + NPZ by default.
4. Optional integrations fail clearly when their backend is absent.
5. Reproducibility metadata, provenance, units, and domain identity are first-class.

## License

This project is licensed under the [MIT License](LICENSE).
