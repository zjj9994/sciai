"""Molecular graph data and a dependency-free SMILES subset parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sciai.core import GraphData, ScienceMetadata
from sciai.exceptions import ValidationError

_ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
# Supported SMILES tokens: organic-subset atoms (aliphatic uppercase, aromatic
# lowercase), branches, bond symbols, and single ring-closure digits.
_TOKEN = re.compile(r"Cl|Br|[BCNOFPSI]|[cnops]|\(|\)|=|#|-|\d")
_BOND_ORDER = {"-": 1.0, "=": 2.0, "#": 3.0}
_AROMATIC = frozenset({"b", "c", "n", "o", "p", "s"})


@dataclass(kw_only=True)
class Molecule(GraphData):
    """A molecular graph with atom symbols and bond orders."""

    symbols: tuple[str, ...]
    smiles: str | None = None
    bond_orders: Any = field(default_factory=lambda: np.empty(0, dtype=float))

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.symbols) != self.values.shape[0]:
            raise ValidationError("There must be one element symbol per molecular graph node")
        unknown = set(self.symbols) - set(_ATOMIC_NUMBERS)
        if unknown:
            raise ValidationError(f"Unsupported element symbols: {sorted(unknown)}")
        self.bond_orders = np.asarray(self.bond_orders, dtype=float)
        if self.bond_orders.shape != (self.edge_index.shape[1],):
            raise ValidationError("There must be one bond order per directed edge")

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.asarray([_ATOMIC_NUMBERS[symbol] for symbol in self.symbols], dtype=float)

    @classmethod
    def from_smiles(cls, smiles: str) -> Molecule:
        """Parse a strict, conservative subset of SMILES without external dependencies.

        Supported syntax
        ----------------
        * Common atoms: ``B C N O P S F Cl Br I`` (aliphatic) and
          ``b c n o p s`` (aromatic; only these lowercase elements are aromatic).
        * Branches delimited by balanced ``(`` ``)``.
        * Explicit bonds: ``-`` (single), ``=`` (double), ``#`` (triple).
        * Single-digit ring closures (``1``-``9``) appearing exactly twice.

        Bond orders
        -----------
        * An explicit bond symbol always wins (it also takes priority inside a ring
          closure: ``C=1CCCCC1`` and ``C1=CCCCC1`` both close with a double bond).
        * Without an explicit bond, a bond between two aromatic atoms is the implied
          aromatic bond order of ``1.5``; any other bond is single (``1.0``). A bond
          between an aromatic and an aliphatic atom is therefore single, never ``1.5``.

        Scope and limits
        -----------------
        This parser is intentionally conservative. It does **not** perform full
        valence, stereochemistry, isotope, or charge validation, and it does **not**
        pretend to be RDKit. Unsupported syntax -- bracket atoms, charges, isotopes,
        stereochemical markers, multi-digit ``%nn`` ring closures, disconnected
        components, or any other token -- raises :class:`ValidationError`. The
        following are also rejected: a bond before the first atom or after the last
        atom, duplicate bonds, self-loops, unbalanced or empty branches, and
        conflicting ring-closure bond orders.

        Install an RDKit-backed domain plugin for full SMILES semantics.
        """
        compact = smiles.strip()
        if not compact:
            raise ValidationError("SMILES string cannot be empty")

        tokens = _TOKEN.findall(compact)
        if "".join(tokens) != compact:
            raise ValidationError(
                "Unsupported SMILES syntax. The core parser supports common atoms, "
                "branches, bond symbols, and single-digit ring closures; install an "
                "RDKit domain plugin for full SMILES semantics."
            )

        symbols: list[str] = []
        aromatic_flags: list[bool] = []
        edges: list[tuple[int, int, float]] = []
        seen_edges: set[frozenset[int]] = set()
        branches: list[list[Any]] = []  # stack of [branch_entry_atom, had_atom]
        rings: dict[str, tuple[int, float | None]] = {}
        current: int | None = None
        pending_bond = 1.0
        pending_explicit = False

        def add_edge(source: int, target: int, order: float) -> None:
            if source == target:
                raise ValidationError("SMILES ring closure forms a self-loop")
            key = frozenset((source, target))
            if key in seen_edges:
                raise ValidationError(
                    f"Duplicate bond between atoms {source} and {target}"
                )
            seen_edges.add(key)
            edges.append((source, target, order))

        for token in tokens:
            if token in _BOND_ORDER:
                if current is None:
                    raise ValidationError("A bond symbol cannot appear before the first atom")
                pending_bond = _BOND_ORDER[token]
                pending_explicit = True
            elif token == "(":
                if current is None:
                    raise ValidationError("SMILES branch cannot start before an atom")
                branches.append([current, False])
            elif token == ")":
                if not branches:
                    raise ValidationError("Unbalanced SMILES branch: ')' without '('")
                _entry, had_atom = branches.pop()
                if not had_atom:
                    raise ValidationError("SMILES branch cannot be empty")
                current = _entry
            elif token.isdigit():
                if current is None:
                    raise ValidationError("Ring closure cannot appear before an atom")
                explicit = pending_bond if pending_explicit else None
                if token not in rings:
                    rings[token] = (current, explicit)
                else:
                    first_atom, first_explicit = rings.pop(token)
                    second_explicit = pending_bond if pending_explicit else None
                    if (
                        first_explicit is not None
                        and second_explicit is not None
                        and first_explicit != second_explicit
                    ):
                        raise ValidationError(
                            f"Conflicting ring closure bond order for digit {token!r}"
                        )
                    if first_explicit is not None:
                        order = first_explicit
                    elif second_explicit is not None:
                        order = second_explicit
                    else:
                        order = (
                            1.5
                            if aromatic_flags[first_atom] and aromatic_flags[current]
                            else 1.0
                        )
                    add_edge(first_atom, current, order)
                pending_bond = 1.0
                pending_explicit = False
            else:
                is_aromatic = token in _AROMATIC
                index = len(symbols)
                symbols.append(token.capitalize())
                aromatic_flags.append(is_aromatic)
                if current is not None:
                    if pending_explicit:
                        order = pending_bond
                    else:
                        order = 1.5 if aromatic_flags[current] and is_aromatic else 1.0
                    add_edge(current, index, order)
                if branches:
                    branches[-1][1] = True
                current = index
                pending_bond = 1.0
                pending_explicit = False

        if branches:
            raise ValidationError("Unbalanced SMILES branch: '(' without ')'")
        if rings:
            raise ValidationError("Unclosed SMILES ring closure")
        if pending_explicit:
            raise ValidationError("A bond symbol cannot follow the final atom")

        directed: list[tuple[int, int]] = []
        orders: list[float] = []
        for source, target, order in edges:
            directed.extend(((source, target), (target, source)))
            orders.extend((order, order))
        edge_index: np.ndarray = (
            np.asarray(directed, dtype=np.int64).T
            if directed
            else np.empty((2, 0), dtype=np.int64)
        )
        atomic_numbers: np.ndarray = np.asarray(
            [_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=float
        )
        node_features = np.column_stack(
            [atomic_numbers, atomic_numbers / max(float(atomic_numbers.max()), 1.0)]
        )
        return cls(
            values=node_features,
            edge_index=edge_index,
            edge_features=np.asarray(orders, dtype=float).reshape(-1, 1),
            symbols=tuple(symbols),
            smiles=compact,
            bond_orders=np.asarray(orders, dtype=float),
            metadata=ScienceMetadata(
                name=compact,
                quantity="molecular-graph",
                domain="molecule",
            ),
        )

    def descriptors(self) -> np.ndarray:
        """Return deterministic low-cost descriptors for baseline models."""
        heavy_atoms = float(sum(symbol != "H" for symbol in self.symbols))
        hetero_atoms = float(sum(symbol not in {"C", "H"} for symbol in self.symbols))
        undirected_bonds = float(self.edge_index.shape[1] / 2)
        total_bond_order = float(np.sum(self.bond_orders) / 2)
        atomic_number_sum = float(np.sum(self.atomic_numbers))
        return np.asarray(
            [heavy_atoms, hetero_atoms, undirected_bonds, total_bond_order, atomic_number_sum],
            dtype=float,
        )
