from functools import partial
from typing import Any

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    AttrComparisonSpec,
    EntityReferenceComparisonConfig,
)
from matchescu.typing import DataSource, Record, EntityReference


class RecordLinkageDataSet:
    __TARGET_COL = "y"

    def __init__(
        self,
        left: DataSource[Record],
        right: DataSource[Record],
        ground_truth: set[tuple[Any, Any]],
    ) -> None:
        self.__extract_left = EntityReferenceExtraction(left, lambda ref: ref[0])
        self.__extract_right = EntityReferenceExtraction(right, lambda ref: ref[0])
        self.__true_matches = ground_truth
        self.__comparison_data = None

    @property
    def target_vector(self) -> pl.DataFrame:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return self.__comparison_data[self.__TARGET_COL]

    @property
    def feature_matrix(self) -> pl.DataFrame:
        if self.__comparison_data is None:
            raise ValueError("comparison matrix was not computed")
        return self.__comparison_data.drop([self.__TARGET_COL])

    @staticmethod
    def __with_col_suffix(
        extract: EntityReferenceExtraction, suffix: str
    ) -> pl.DataFrame:
        df = pl.DataFrame(extract())
        return df.rename({key: f"{key}{suffix}" for key in df.columns})

    @staticmethod
    def __compare_attr_values(
        left_ref: EntityReference,
        right_ref: EntityReference,
        config: AttrComparisonSpec,
    ) -> int:
        a = left_ref[config.left_ref_key]
        b = right_ref[config.right_ref_key]
        return config.match_strategy(a, b)

    def __compare_entity_references(
        self,
        comparison_row: tuple,
        midpoint: int,
        config: EntityReferenceComparisonConfig,
    ) -> tuple:
        left_side = comparison_row[:midpoint]
        right_side = comparison_row[midpoint:]
        left_id = self.__extract_left.identify(left_side)
        right_id = self.__extract_right.identify(right_side)
        result = {
            spec.label: self.__compare_attr_values(left_side, right_side, spec)
            for spec in config.specs
        }
        result[self.__TARGET_COL] = int((left_id, right_id) in self.__true_matches)
        return (result,)  # need to return a tuple

    def create_comparison_matrix(
        self, config: EntityReferenceComparisonConfig
    ) -> "RecordLinkageDataSet":
        left = self.__with_col_suffix(self.__extract_left, "_left")
        right = self.__with_col_suffix(self.__extract_right, "_right")
        cross_product = left.join(right, how="cross")
        compare_entity_references = partial(
            self.__compare_entity_references, midpoint=len(left.columns), config=config
        )
        dtype = {spec.label: pl.UInt8 for spec in config.specs}
        self.__comparison_data = cross_product.map_rows(
            compare_entity_references, pl.Struct(dtype)
        ).unnest("column_0")
        return self
