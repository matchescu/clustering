import pytest

from matchescu.matching.similarity import Similarity
from matchescu.matching.attribute import (
    TernaryResult,
    TernarySimilarityMatchOnThreshold,
    BinarySimilarityMatchOnThreshold,
    BinaryResult,
)


class SimilarityStub(Similarity):
    def __init__(self, sim_score: float = 0.5):
        self._sim_score = sim_score

    def _compute_similarity(self, _, __) -> float:
        return self._sim_score


@pytest.mark.parametrize(
    "match_strategy,expected",
    [
        (TernarySimilarityMatchOnThreshold, TernaryResult.NoComparisonData),
        (BinarySimilarityMatchOnThreshold, BinaryResult.Negative),
    ],
)
def test_no_comparison_data(match_strategy, expected):
    is_match = match_strategy(SimilarityStub())

    assert is_match(None, None) == expected


@pytest.mark.parametrize(
    "match_strategy, similarity, threshold, expected",
    [
        (TernarySimilarityMatchOnThreshold, 0, 0.01, TernaryResult.NonMatch),
        (TernarySimilarityMatchOnThreshold, 1, 1, TernaryResult.Match),
        (TernarySimilarityMatchOnThreshold, 0.5, 0.49, TernaryResult.Match),
        (BinarySimilarityMatchOnThreshold, 0, 0.01, BinaryResult.Negative),
        (BinarySimilarityMatchOnThreshold, 1, 1, BinaryResult.Positive),
        (BinarySimilarityMatchOnThreshold, 0.5, 0.49, BinaryResult.Positive),
    ],
)
def test_value_similarity_match(match_strategy, similarity, threshold, expected):
    is_match = match_strategy(SimilarityStub(similarity), threshold)

    assert is_match("can pass any value", "with stubbed similarity") == expected
