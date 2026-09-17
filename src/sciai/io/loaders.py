"""Extensible scientific file loading with explicit format detection."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np

from sciai.core import ScalarField, ScienceData, ScienceMetadata
from sciai.exceptions import ValidationError
from sciai.registry import data_loader_registry


class DataLoader(ABC):
    """Adapter from an external scientific format to ScienceData."""

    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path, **kwargs: Any) -> ScienceData:
        """Load one file."""


class _LoaderHelpers:
    """Shared keyword-argument and array-safety checks for concrete loaders."""

    @staticmethod
    def reject_unknown(kwargs: dict[str, Any], known: set[str]) -> None:
        extra = set(kwargs) - known
        if extra:
            raise ValidationError(f"Unsupported loader arguments: {sorted(extra)}")

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        try:
            array = np.load(path, allow_pickle=False)
        except ValueError as exc:
            raise ValidationError(
                f"Refusing to load object-dtype array from {path}: {exc}"
            ) from exc
        if getattr(array, "dtype", None) is not None and array.dtype.kind == "O":
            raise ValidationError(f"Refusing to load object-dtype array from {path}")
        return np.asarray(array)


class NpyLoader(DataLoader):
    extensions = (".npy",)

    def load(self, path: Path, **kwargs: Any) -> ScienceData:
        _LoaderHelpers.reject_unknown(kwargs, {"unit", "metadata"})
        array = _LoaderHelpers.load_npy(path)
        return ScalarField(
            values=array,
            unit=kwargs.get("unit", "1"),
            metadata=kwargs.get("metadata", ScienceMetadata(name=path.stem)),
        )


class NpzLoader(DataLoader):
    extensions = (".npz",)

    def load(self, path: Path, **kwargs: Any) -> ScienceData:
        _LoaderHelpers.reject_unknown(kwargs, {"key", "unit", "metadata"})
        key = kwargs.get("key")
        try:
            with np.load(path, allow_pickle=False) as archive:
                names = archive.files
                if key is None:
                    if len(names) != 1:
                        raise ValidationError(
                            f"NPZ archive contains {names}; specify key=... to choose an array"
                        )
                    key = names[0]
                if key not in names:
                    raise ValidationError(
                        f"NPZ key {key!r} not found; available: {names}"
                    )
                array = archive[key]
        except ValueError as exc:
            raise ValidationError(
                f"Refusing to load object-dtype array from {path}: {exc}"
            ) from exc
        if getattr(array, "dtype", None) is not None and array.dtype.kind == "O":
            raise ValidationError(f"Refusing to load object-dtype array from {path}")
        return ScalarField(
            values=array,
            unit=kwargs.get("unit", "1"),
            metadata=kwargs.get("metadata", ScienceMetadata(name=path.stem)),
        )


class CsvLoader(DataLoader):
    extensions = (".csv",)

    def load(self, path: Path, **kwargs: Any) -> ScienceData:
        _LoaderHelpers.reject_unknown(kwargs, {"delimiter", "skip_header", "unit", "metadata"})
        delimiter = kwargs.get("delimiter", ",")
        skip_header = kwargs.get("skip_header", True)
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream, delimiter=delimiter))
        if not rows:
            raise ValidationError(f"CSV file {path} is empty")
        if skip_header:
            rows = rows[1:]
        if not rows:
            raise ValidationError(
                f"CSV file {path} contains only a header with no data rows"
            )
        try:
            values = np.asarray(rows, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                f"CSV file {path} contains non-numeric values"
            ) from exc
        if values.ndim != 2:
            raise ValidationError(f"CSV file {path} has inconsistent row lengths")
        return ScalarField(
            values=values,
            unit=kwargs.get("unit", "1"),
            metadata=kwargs.get("metadata", ScienceMetadata(name=path.stem)),
        )


def register_builtin_loaders() -> None:
    for loader_type in (NpyLoader, NpzLoader, CsvLoader):
        for extension in loader_type.extensions:
            if not data_loader_registry.contains(extension):
                data_loader_registry.register(extension, loader_type)


def load_data(path: str | Path, *, format: str | None = None, **kwargs: Any) -> ScienceData:
    """Detect a registered format and return a standardized ScienceData object."""
    register_builtin_loaders()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    format_key = (format or source.suffix).lower()
    if not format_key.startswith("."):
        format_key = f".{format_key}"
    loader_type = data_loader_registry.get(format_key)
    return cast(ScienceData, loader_type().load(source, **kwargs))
