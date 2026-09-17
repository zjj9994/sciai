"""Dataset abstractions shared across scientific domains."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload

from sciai.exceptions import ValidationError

T = TypeVar("T")


@dataclass(frozen=True)
class DatasetInfo:
    """Machine-readable dataset provenance and scientific semantics."""

    name: str
    domain: str = "general"
    version: str = "0"
    license: str | None = None
    citation: str | None = None
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


class ScienceDataset(Sequence[T], Generic[T]):
    """Indexable scientific dataset with provenance and a collate contract."""

    def __init__(
        self,
        records: Sequence[T],
        *,
        info: DatasetInfo,
        collate_fn: Callable[[list[T]], Any] | None = None,
    ) -> None:
        if len(records) == 0:
            raise ValidationError("A ScienceDataset must contain at least one record")
        self._records = tuple(records)
        self.info = info
        self._collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        return self._records[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._records)

    def collate(self, records: list[T]) -> Any:
        if self._collate_fn is not None:
            return self._collate_fn(records)
        return records
