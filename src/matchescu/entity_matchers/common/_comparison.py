from dataclasses import dataclass
from typing import Any, Type, Iterable

from matchescu.entity_matchers.attribute_matching import AttrMatchCallable, FSMatch
from matchescu.entity_matchers.similarity import (
    ExactMatch,
    Jaro,
    Jaccard,
    JaroWinkler,
    Levenshtein,
)


@dataclass
class BinaryComparisonSpec:
    label: str
    left_ref_key: int | str
    right_ref_key: int | str
    match_strategy: AttrMatchCallable


class FSComparison:
    def __init__(self, *args: BinaryComparisonSpec):
        self._specs = list(*args)

    @staticmethod
    def _new_spec(
        similarity_type: Type,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold: float,
        *args: Any
    ) -> BinaryComparisonSpec:
        return BinaryComparisonSpec(
            label=label,
            left_ref_key=left_key,
            right_ref_key=right_key,
            match_strategy=FSMatch(similarity_type(*args), threshold),
        )

    def exact(
        self, label: str, left_key: int | str, right_key: int | str
    ) -> "FSComparison":
        self._specs.append(
            self._new_spec(ExactMatch, label, left_key, right_key, 1)
        )
        return self

    def jaro(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "FSComparison":
        self._specs.append(
            self._new_spec(
                Jaro, label, left_key, right_key, threshold, ignore_case
            )
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
    ) -> "FSComparison":
        self._specs.append(self._new_spec(
                Jaccard, label, left_key, right_key, threshold, ignore_case, gram_size
            ))
        return self

    def jaro_winkler(
        self,
        label: str,
        left_key: int | str,
        right_key: int | str,
        threshold=0.5,
        ignore_case: bool = False,
    ) -> "FSComparison":
        self._specs.append(
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
    ) -> "FSComparison":
        self._specs.append(
            self._new_spec(
                Levenshtein, label, left_key, right_key, threshold, ignore_case
            )
        )
        return self

    @property
    def specs(self) -> Iterable[BinaryComparisonSpec]:
        return self._specs
