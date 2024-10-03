from abc import abstractmethod
from typing import Any

from matchescu.matching.attribute._match_result import (
    MatchResult,
    FSMatchResult,
    NBMatchResult,
)
from matchescu.matching.similarity._common import Similarity


class SimilarityThresholdMatch:
    def __init__(
        self, similarity: Similarity, similarity_threshold: float = 0.5
    ) -> None:
        self._threshold = similarity_threshold
        self._compute_similarity = similarity
        self._similarity: float | None = None

    @property
    def similarity(self) -> float | None:
        return self._similarity

    @abstractmethod
    def _indeterminate(self) -> MatchResult:
        pass

    @abstractmethod
    def _match(self) -> MatchResult:
        pass

    @abstractmethod
    def _non_match(self) -> MatchResult:
        pass

    def __call__(self, a: Any, b: Any) -> MatchResult:
        if a is None and b is None:
            return self._indeterminate()

        self._similarity = self._compute_similarity(a, b)
        if self._similarity < self._threshold:
            return self._non_match()
        return self._match()


class FSMatch(SimilarityThresholdMatch):
    def _indeterminate(self) -> MatchResult:
        return FSMatchResult.NoComparisonData

    def _match(self) -> MatchResult:
        return FSMatchResult.Match

    def _non_match(self) -> MatchResult:
        return FSMatchResult.NonMatch


class NBMatch(SimilarityThresholdMatch):
    def _indeterminate(self) -> MatchResult:
        return NBMatchResult.Negative

    def _match(self) -> MatchResult:
        return NBMatchResult.Positive

    def _non_match(self) -> MatchResult:
        return NBMatchResult.Negative
