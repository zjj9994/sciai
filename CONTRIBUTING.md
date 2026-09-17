# Contributing

`sciai` is in a pre-alpha architecture phase. Changes should strengthen stable scientific contracts rather than add unverified catalog breadth.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/sciai
pytest --cov=sciai
python -m build
```

## Domain contributions

Start an independently distributable package:

```bash
sciai scaffold-domain <domain-name> --destination ./plugins
```

A production domain plugin should provide:

- scientific data semantics and validation;
- domain operators with numerical tests;
- model metadata and reproducible recipes;
- dataset provenance, license, citation, and checksums;
- simulator adapter tests where applicable;
- explicit optional dependencies;
- no claims of SOTA status without a pinned benchmark and reproducible evidence.

## Core acceptance criteria

- No mandatory dependency on a deep-learning framework or commercial solver.
- Unit, dimension, coordinate, topology, and artifact invariants are checked at boundaries.
- Serialization does not execute untrusted code by default.
- New public behavior includes tests and compatibility notes.
- External software names are described as integrations only when an adapter is tested.

## Commit scope

Keep commits focused and do not combine broad formatting changes with behavioral work. Before contributing publicly, the project needs an explicit license, code of conduct, governance policy, and security reporting channel.
