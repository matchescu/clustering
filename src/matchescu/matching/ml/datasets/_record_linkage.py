import itertools
from functools import partial
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    AttrComparisonSpec,
    EntityReferenceComparisonConfig,
)
from matchescu.typing import DataSource, Record


class RecordLinkageDataSet:
    def __init__(
        self,
        left: DataSource[Record],
        right: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
    ) -> None:
        self.__extract_left = EntityReferenceExtraction(left, lambda ref: ref[0])
        self.__extract_right = EntityReferenceExtraction(right, lambda ref: ref[0])
        self.__true_matches = ground_truth
        self._y = None
        self._x = None

    def _is_true_match(self, ids: tuple | dict) -> int:
        if isinstance(ids, dict):
            ids = tuple(ids.values())
        return 1 if ids in self.__true_matches else 0

    def _compute_target_vector(self) -> np.array:
        tuples = list(
            itertools.product(
                self.__extract_left.entity_ids(),
                self.__extract_right.entity_ids(),
            )
        )
        df = pl.DataFrame(
            tuples, {"id_left": pl.Int64, "id_right": pl.Int64}, orient="row"
        )
        return (
            df.lazy()
            .with_columns(
                [
                    pl.struct(["id_left", "id_right"])
                    .map_elements(self._is_true_match, pl.Int8)
                    .alias("y")
                ]
            )
            .select("y")
            .collect()
            .to_numpy()
        )

    @property
    def target_vector(self):
        if self._y is None:
            self._y = self._compute_target_vector()
        return self._y

    @staticmethod
    def __with_col_suffix(
        extract: EntityReferenceExtraction, suffix: str
    ) -> pl.DataFrame:
        df = pl.DataFrame(extract())
        return df.rename({key: f"{key}{suffix}" for key in df.columns})

    @staticmethod
    def __compare_attributes(
        left_row: tuple, right_row: tuple, config: AttrComparisonSpec
    ) -> int:
        a = left_row[config.left_ref_key]
        b = right_row[config.right_ref_key]
        return config.match_strategy(a, b)

    @classmethod
    def __compare_entity_references(
        cls,
        comparison_row: tuple,
        midpoint: int,
        config: EntityReferenceComparisonConfig,
    ) -> tuple:
        left_side = comparison_row[:midpoint]
        right_side = comparison_row[midpoint:]
        result = {
            spec.label: cls.__compare_attributes(left_side, right_side, spec)
            for spec in config.specs
        }
        return (result,)  # need to return a tuple

    def compute_feature_matrix(
        self, config: EntityReferenceComparisonConfig
    ) -> pl.DataFrame:
        left = self.__with_col_suffix(self.__extract_left, "_left")
        right = self.__with_col_suffix(self.__extract_right, "_right")

        cross_product = left.join(right, how="cross")
        compare_entity_references = partial(
            self.__compare_entity_references, midpoint=len(left.columns), config=config
        )
        dtype = {spec.label: pl.UInt8 for spec in config.specs}
        return cross_product.map_rows(
            compare_entity_references, pl.Struct(dtype)
        ).unnest("column_0")
