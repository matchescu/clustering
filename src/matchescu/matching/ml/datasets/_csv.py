from os import PathLike
from typing import Iterable, Iterator

import polars as pl

from matchescu.typing import Trait, Record


class CsvDataSource:
    def __init__(self, name: str, traits: Iterable[Trait]):
        self.name = name
        self.traits = traits
        self._df = None

    def read_csv(self, path: str | PathLike) -> "CsvDataSource":
        self._df = pl.read_csv(path, ignore_errors=True)
        return self

    def __iter__(self) -> Iterator[Record]:
        if self._df is None:
            return iter([])
        return self._df.iter_rows()

    def __len__(self) -> int:
        if self._df is None:
            return 0
        return self._df.shape[0]
