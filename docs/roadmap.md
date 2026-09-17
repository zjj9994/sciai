# Roadmap

## Phase 0: foundation (`0.1.x`)

- Stabilize ontology, data, model, trainer, metric, plugin, pipeline, workflow, and simulation contracts.
- Add schema/version migration tests and public API compatibility policy.
- Select project license, governance model, contribution process, and security policy.
- Publish architectural decision records and benchmark methodology.

## Phase 1: production backends (`0.2.x`)

- Separate `sciai-torch`, `sciai-jax`, and `sciai-mindspore` trainer/model bridges.
- Mixed precision, gradient accumulation, checkpoint resumption, distributed data parallelism.
- Experiment lineage, configuration capture, deterministic controls, and environment manifests.
- Array protocol support for NumPy, DLPack, and lazy arrays.

## Phase 2: scientific data hub (`0.3.x`)

- Format plugins for HDF5, NetCDF, Zarr, VTK, CIF/PDB, DICOM/NIfTI, FITS, and trajectory formats.
- Download cache with hashes, licenses, citations, split manifests, and offline mode.
- Unit normalization, coordinate alignment, quality checks, and provenance-preserving transforms.

A claim of "100+ formats" or "50+ datasets" will only be made after automated compatibility tests and provenance records exist for every listed item.

## Phase 3: official domain packs (`0.4.x` onward)

Deliver independent packages with domain maintainers and benchmark gates:

- molecular and chemistry
- materials
- PDE and mathematical physics
- fluids, solids, thermal, and electromagnetics
- climate, atmosphere, ocean, hydrology, and seismology
- life science, drug discovery, genomics, and medical imaging
- astronomy and space science

Each official pack must include tested data adapters, domain operators, reproducible model recipes, dataset manifests, and at least one simulator bridge or documented reason none applies.

## Phase 4: model and dataset Hub

- Signed immutable manifests, checksums, semantic versions, model cards, dataset cards, citations, and licenses.
- Trust policy for custom code; no implicit remote code execution.
- Conversion tools across backend-native checkpoint formats.
- Reproducibility scorecards based on executable recipes.

## Phase 5: co-simulation and scientific workflows

- Time synchronization, implicit/explicit coupling schemes, interpolation across meshes, conservation checks, convergence criteria, rollback, and fault recovery.
- OpenFOAM, FEniCS, LAMMPS, GROMACS, and other open integrations first.
- Vendor adapters only under valid licensing and integration terms.
- Visual workflow editor backed by the same serialized DAG, not a separate runtime.

## Release gates

Every milestone requires:

1. public API tests and migration notes;
2. numerical validation against analytical or trusted reference solutions;
3. reproducible environment and data provenance;
4. security review for loaders and remote artifacts;
5. benchmark evidence for performance claims;
6. documentation that distinguishes bundled, optional, and planned capabilities.
