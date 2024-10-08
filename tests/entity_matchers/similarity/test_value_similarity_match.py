from typing import Any

import pytest

from matchescu.entity_matchers.similarity import Similarity
from matchescu.entity_matchers.attribute_matching import FSMatch, FSMatchResult


class SimilarityStub(Similarity):
    def __init__(self, sim_score: float = 0.5):
        self._sim_score = sim_score

    def _compute_similarity(self, _, __) -> float:
        return self._sim_score


def test_no_comparison_data():
    is_match = FSMatch(SimilarityStub())

    assert is_match(None, None) == FSMatchResult.NoComparisonData


@pytest.mark.parametrize(
    "similarity, threshold, expected",
    [
        (0, 0.01, FSMatchResult.NonMatch),
        (1, 1, FSMatchResult.Match),
        (0.5, 0.49, FSMatchResult.Match),
    ],
)
def test_value_similarity_match(similarity, threshold, expected):
    is_match = FSMatch(SimilarityStub(similarity), threshold)

    assert is_match("any value", "stub method") == expected
