import pytest

from matchescu.clustering._mcl import MarkovClustering
from pyresolvemetrics import twi
from tests.testutil import is_partition_over


@pytest.fixture
def mcl(all_refs):
    return MarkovClustering(all_refs)


def test_on_chain(mcl, all_refs, chain_digraph):
    clusters = mcl(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1, "expected chain to be clustered"


def test_on_ring(mcl, all_refs, ring_digraph):
    clusters = mcl(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) >= 3, "expected ring to explode"


def test_clique(mcl, all_refs, clique_digraph):
    clusters = mcl(clique_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1, f"Clique should be 1 cluster, got {len(clusters)}"


@pytest.mark.parametrize(
    "all_refs",
    [
        [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
        ]
    ],
    indirect=True,
)
def test_ring_with_cliques(mcl, all_refs, ring_with_cliques_digraph):
    clusters = mcl(ring_with_cliques_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) > 3, "Expected ring with cliques to explode"


@pytest.mark.skip(reason="only run this locally - not in CI")
def test_partitioning_on_real_data(
    matcher_mock, dataset_refs, dataset_ground_truth, dataset_fwd_graph
):
    algorithm = MarkovClustering(dataset_refs, threshold=0.4)

    actual = algorithm(dataset_fwd_graph)

    assert is_partition_over(dataset_refs, actual)
    score = twi(dataset_ground_truth, actual)
    assert 0 <= score <= 1
