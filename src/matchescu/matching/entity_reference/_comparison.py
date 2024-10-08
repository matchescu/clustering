from abc import abstractmethod
from typing import Any, Type, Iterable

from matchescu.matching.attribute import TernarySimilarityMatchOnThreshold
from matchescu.matching.attribute._match import SimilarityMatch
from matchescu.matching.entity_reference._attr_spec import AttrComparisonSpec
from matchescu.matching.similarity import (
    ExactMatch,
    Jaro,
    Jaccard,
    JaroWinkler,
    Levenshtein, Similarity,
)


class EntityReferenceComparisonConfig:
    def __init__(self):
        self.__specs = []

    @classmethod
    @abstractmethod
    def _new_similarity_threshold_match_strategy(cls, similarity: Similarity, *args) -> SimilarityMatch:
        pass

    @classmethod
    def _new_spec(
        cls,
        similarity_type: Type,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float,
        *args: Any
    ) -> AttrComparisonSpec:
        return AttrComparisonSpec(
            label=label,
            left_ref_key=left_key,
            right_ref_key=right_key,
            match_strategy=cls._new_similarity_threshold_match_strategy(similarity_type(*args), threshold),
        )

    def exact(
        self, label: str, left_key: int | str, right_key: int | str
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(self._new_spec(ExactMatch, label, left_key, right_key, 1))
        return self

    def jaro(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(Jaro, label, left_key, right_key, threshold, ignore_case)
        )
        return self

    def jaccard(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float = 0.5,
        gram_size: int | None = None,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                Jaccard, label, left_key, right_key, threshold, ignore_case, gram_size
            )
        )
        return self

    def jaro_winkler(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                JaroWinkler, label, left_key, right_key, threshold, ignore_case
            )
        )
        return self

    def levenshtein(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "EntityReferenceComparisonConfig":
        self.__specs.append(
            self._new_spec(
                Levenshtein, label, left_key, right_key, threshold, ignore_case
            )
        )
        return self

    @property
    def specs(self) -> Iterable[AttrComparisonSpec]:
        return self.__specs


class FellegiSunterComparison(EntityReferenceComparisonConfig):
    @classmethod
    def _new_similarity_threshold_match_strategy(cls, similarity: Similarity, *args) -> SimilarityMatch:
        return TernarySimilarityMatchOnThreshold(similarity, *args)
