"""Command-line utilities for inspecting and extending sciai."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path

from sciai import __version__
from sciai.domains import DOMAIN_CATALOG, load_builtin_plugins
from sciai.registry import domain_registry
from sciai.scaffold import scaffold_domain


def _catalog(as_json: bool) -> int:
    load_builtin_plugins()
    payload = [
        {
            "key": family.key,
            "name": family.name,
            "status": family.status.value,
            "subdomains": family.subdomains,
        }
        for family in DOMAIN_CATALOG
    ]
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("Scientific domain catalog")
        for family in payload:
            print(
                f"- {family['key']}: {family['status']} "
                f"({len(family['subdomains'])} subdomains)"
            )
        print(f"Loaded plugins: {', '.join(domain_registry.keys())}")
    return 0


def _doctor() -> int:
    optional = ("torch", "jax", "mindspore", "rdkit", "meshio")
    print(f"sciai {__version__}")
    print("core: ready")
    for package in optional:
        print(f"{package}: {'installed' if find_spec(package) else 'not installed'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sciai")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="List scientific domain coverage")
    catalog.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    scaffold = subparsers.add_parser(
        "scaffold-domain", help="Create a separately distributable domain plugin"
    )
    scaffold.add_argument("name")
    scaffold.add_argument("--destination", type=Path, default=Path.cwd())

    subparsers.add_parser("doctor", help="Inspect optional scientific backends")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "catalog":
        return _catalog(arguments.json)
    if arguments.command == "doctor":
        return _doctor()
    if arguments.command == "scaffold-domain":
        created = scaffold_domain(arguments.name, arguments.destination)
        print(created)
        return 0
    raise AssertionError(f"Unhandled command {arguments.command!r}")
