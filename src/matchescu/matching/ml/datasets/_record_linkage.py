from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Hashable

import polars as pl

from matchescu.data import EntityReferenceExtraction
from matchescu.matching.entity_reference import (
    AttrComparisonSpec,
    EntityReferenceComparisonConfig,
)
from matchescu.typing import DataSource, Record, EntityReference


class SampleFactory(metaclass=ABCMeta):
    def __init__(
        self,
        ground_truth: set[tuple[Hashable, Hashable]],
        cmp_config: EntityReferenceComparisonConfig,
        cmp_row_separator: int,
        left_id: Callable[[EntityReference], Hashable],
        right_id: Callable[[EntityReference], Hashable],
        target_col_name: str,
    ):
        self._gt = ground_truth
        self._config = cmp_config
        self._row_sep = cmp_row_separator
        self._left_id = left_id
        self._right_id = right_id
        self._target_col = target_col_name

    @abstractmethod
    def _process_comparison(
        self, left: EntityReference, right: EntityReference
    ) -> dict:
        pass

    def __call__(self, cross_join_row: tuple) -> tuple[dict]:
        left_side = cross_join_row[: self._row_sep]
        right_side = cross_join_row[self._row_sep :]
        result = self._process_comparison(left_side, right_side)
        result[self._target_col] = int(
            (self._left_id(left_side), self._right_id(right_side)) in self._gt
        )
        return (result,)  # need to return a tuple


class AttributeComparison(SampleFactory):
    @staticmethod
    def __compare_attr_values(
        left_ref: EntityReference,
        right_ref: EntityReference,
        config: AttrComparisonSpec,
    ) -> int:
        a = left_ref[config.left_ref_key]
        b = right_ref[config.right_ref_key]
        return config.match_strategy(a, b)

    def _process_comparison(
        self, left: EntityReference, right: EntityReference
    ) -> dict:
        return {
            spec.label: self.__compare_attr_values(left, right, spec)
            for spec in self._config.specs
        }


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

    def create_comparison_matrix(
        self,
        config: EntityReferenceComparisonConfig,
        sample_factory: Callable[[tuple], tuple[dict]] | None = None,
    ) -> "RecordLinkageDataSet":
        left = self.__with_col_suffix(self.__extract_left, "_left")
        right = self.__with_col_suffix(self.__extract_right, "_right")
        cross_product = left.join(right, how="cross")
        sample_factory = sample_factory or AttributeComparison(
            self.__true_matches,
            config,
            len(left.columns),
            self.__extract_left.identify,
            self.__extract_right.identify,
            self.__TARGET_COL,
        )
        self.__comparison_data = cross_product.map_rows(sample_factory).unnest(
            "column_0"
        )
        return self
