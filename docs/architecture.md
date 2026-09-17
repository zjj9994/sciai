# Architecture

## System boundary

`sciai` defines interoperability contracts. It is not another tensor runtime and does not reimplement mature scientific solvers. PyTorch, JAX, MindSpore, FEniCS, OpenFOAM, VASP, and commercial software remain independently versioned systems connected through plugins.

## Layer 1: universal scientific foundation

### Ontology

- `Dimension`: rational exponents over the seven SI base dimensions.
- `Unit`: scale and optional affine offset to SI.
- `Quantity`: value plus unit with checked conversion and arithmetic.
- `CoordinateSystem` and `ReferenceFrame`: coordinate semantics independent of array layout.
- `ScienceMetadata`: name, quantity, domain, coordinate context, and extensible attributes.

### Data structures

All common data structures inherit `ScienceData` and carry unit and coordinate metadata:

1. `ScalarField`
2. `VectorField`
3. `TensorField`
4. `GraphData`
5. `ParticleSystem`
6. `MeshData`
7. `TimeSeries`
8. `Spectrum`

These classes validate topology and array invariants but do not prescribe a storage engine. Future adapters may expose lazy arrays while preserving the contract.

### Model and training contracts

`ScienceModel` owns stable lifecycle methods. The core `ScienceTrainer` is a deterministic NumPy reference implementation proving the protocol. Production autodiff, mixed precision, large-batch, distributed, and accelerator execution belong in trainer plugins.

Model artifacts contain:

```text
model.json   # format version, model type, class name, configuration
weights.npz  # numeric arrays loaded with allow_pickle=False
```

This default does not execute arbitrary serialized Python code.

### Operators

The reference operator library includes finite-difference gradient, divergence, curl, Laplacian, trapezoidal integration, FFT/IFFT, one-level Haar transform, convolution, interpolation, and orthogonal vector transforms. Domain plugins may register higher-order, mesh-aware, differentiable, or accelerator-native variants.

## Layer 2: domain extensions

A `DomainPlugin` declares data types, models, datasets, trainers, operators, task pipelines, simulator adapters, subdomains, and metadata. Installed packages expose it through the `sciai.domains` Python entry-point group.

Registration is explicit and process-local. Normalized keys and conflict rejection prevent one plugin from silently replacing another. Namespaced pipeline keys, such as `molecule/property-prediction`, disambiguate common task names.

Built-in plugins are intentionally narrow:

- `molecule`: molecular graph, conservative SMILES parser, trainable descriptor baseline, property-prediction pipeline, synthetic API-test dataset.
- `fluid`: velocity/pressure field, incompressibility diagnostic, conservation-check pipeline, PINN backend facade.

## Layer 3: applications

- `AutoModel`, `AutoDataset`, `AutoTrainer`: resolve local artifacts and installed providers.
- `pipeline`: task-level preprocess, predict, and postprocess flow.
- `ScienceWorkflow`: cross-domain DAG with explicit input references.
- `CoupledSimulator`: directed simulator graph exchanging named variables at interfaces.

## Dependency rule

Imports point inward:

```text
domain plugins -> application protocols -> universal foundation
external adapters -> stable protocols
universal foundation -X-> any domain or tensor framework
```

The core mandatory dependency is NumPy. Format readers, ML frameworks, cluster systems, experiment trackers, and simulation software are optional packages.

## Design decisions

### Framework-neutral core

A single mandatory deep-learning backend would make every other ecosystem a second-class citizen. Models therefore expose a lifecycle contract, while trainer plugins own autodiff and distributed semantics.

### Explicit capability status

A catalog entry means the domain has a stable integration surface, not that every model, dataset, or simulator is already shipped. Status values distinguish implemented references from plugin-ready contracts and roadmap work.

### Adapters over shell commands

Simulation coupling uses typed adapter methods (`initialize`, `step`, `read`, `write`, `finalize`). Native adapters can use files, sockets, MPI, gRPC, or vendor APIs internally without changing orchestration code.

### No automatic remote code execution

Auto loading does not download and import arbitrary repositories. Remote hubs will require signed manifests, checksums, trust policies, and isolated custom-code handling before being added.
