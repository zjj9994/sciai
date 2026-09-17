# Capability Matrix

The matrix describes the `0.1.0a1` repository. "Plugin-ready" means the universal contract and extension point exist; it does not claim a bundled integration.

| Area | Status | Current implementation | Next integration boundary |
|---|---|---|---|
| Units and dimensions | Available | SI base dimensions, derived units, affine temperature conversion | Pint interoperability, uncertainty propagation |
| Coordinates and frames | Available | Axis units, handedness, origin, orientation | CRS/geodesy and frame transforms |
| Eight scientific data types | Available | Eager NumPy reference containers with validation | Lazy/Dask/Xarray and device-array adapters |
| Mathematical operators | Available | Finite differences, FFT, Haar, convolution, interpolation | Autodiff, mesh-aware, equivariant operators |
| Unified model API | Available | Lifecycle and safe artifacts | Torch/JAX/MindSpore bridge packages |
| Unified trainer | Available (reference) | Deterministic NumPy training, SGD/Adam, schedules | AMP, distributed, multiscale, SLURM |
| Metrics | Available | MAE, RMSE, R2, relative error | Domain metric packs and uncertainty calibration |
| Scientific loaders | Available (reference) | NPY, NPZ, numeric CSV | HDF5, NetCDF, VTK, DICOM, FITS, CIF, PDB, trajectories |
| Auto APIs | Available | Explicit local/registered model, dataset, trainer resolution | Signed remote Hub manifests and cache |
| Task pipelines | Available | Molecular property prediction, fluid conservation check | Domain task packs |
| Workflow engine | Available | Cross-domain DAG execution and serialization view | Visual editor, caching, retries, provenance |
| Multiphysics | Available (reference) | Directed variable exchange and in-memory adapter | MPI/socket/co-simulation and convergence schemes |
| Molecule domain | Available (reference) | Graph, SMILES subset, trainable baseline | RDKit, MACE, NequIP, SchNet, DimeNet |
| Fluid domain | Available (reference) | Flow field, divergence residual, PINN facade | OpenFOAM, Fluent, FEniCS, neural operators |
| Other 5 domain families | Plugin-ready | Catalog, data foundation, extension protocol | Official domain packages |
| Reproducibility | Partial | Deterministic trainer seed, artifact config, dataset metadata | environment lock, lineage, experiment store |
| Visualization | Roadmap | Data summaries only | 2D/3D field, graph, particle, spectral viewers |
| Paper export | Roadmap | None | publication figures and LaTeX/CSV tables |
| Cluster scheduling | Plugin-ready | Trainer/config boundary | SLURM, Kubernetes, Ray adapters |

## Domain families

| Family | Subdomains | Repository status |
|---|---:|---|
| Mathematical and physical sciences | 8 | Plugin-ready |
| Chemistry and molecular sciences | 6 | Molecular reference slice available |
| Materials science and engineering | 6 | Plugin-ready |
| Life sciences and medicine | 6 | Plugin-ready |
| Earth and environmental sciences | 7 | Plugin-ready |
| Engineering sciences and technology | 7 | Fluid reference slice available |
| Astronomy and space sciences | 5 | Plugin-ready |

Commercial products listed as integration targets require valid local installations, licenses, and vendor-compliant adapters. `sciai` does not redistribute them.
