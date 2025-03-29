import pytest

from matchescu.typing import EntityReferenceIdentifier as RefId
from matchescu.clustering._tc import TransitiveClosure


@pytest.fixture
def tc(comparison_space):
    return TransitiveClosure(comparison_space)


@pytest.mark.parametrize("matches, expected_length",[
    ([
        (RefId("a", "test"), RefId("b", "test")),
        (RefId("b", "test"), RefId("c", "test")),
        (RefId("c", "test"), RefId("d", "test")),
        (RefId("d", "test"), RefId("a", "test")),
    ], 1),
    ([
         (RefId("a", "test"), RefId("b", "test")),
         (RefId("b", "test"), RefId("c", "test")),
         (RefId("c", "test"), RefId("d", "test")),
         (RefId("c", "test"), RefId("a", "test")),
     ], 2),
    ([
         (RefId("a", "test"), RefId("b", "test")),
         (RefId("b", "test"), RefId("c", "test")),
         (RefId("c", "test"), RefId("d", "test")),
         (RefId("b", "test"), RefId("a", "test")),
     ], 3),
    ([
         (RefId("a", "test"), RefId("b", "test")),
         (RefId("b", "test"), RefId("c", "test")),
         (RefId("c", "test"), RefId("d", "test")),
     ], 4),
])
def test_single_cluster_on_default_data(tc, matches, expected_length):
    clusters = tc(matches)

    assert len(clusters) == expected_length
