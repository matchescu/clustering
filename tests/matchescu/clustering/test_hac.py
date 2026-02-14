import pytest

from matchescu.clustering._hac import HierarchicalAgglomerativeClustering
from pyresolvemetrics import twi
from testutil import is_partition_over


@pytest.fixture
def hac(all_refs):
    return HierarchicalAgglomerativeClustering(all_refs)


def test_on_chain(hac, all_refs, chain_digraph):
    clusters = hac(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1, "expected chain to be preserved"


def test_on_ring(hac, all_refs, ring_digraph):
    clusters = hac(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1, "expected ring to be preserved"


def test_clique(hac, all_refs, clique_digraph):
    clusters = hac(clique_digraph)

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
def test_ring_with_cliques(hac, all_refs, ring_with_cliques_digraph):
    clusters = hac(ring_with_cliques_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1, "Expected everything to be clustered together"


@pytest.mark.skip(reason="only run this locally - not in CI")
def test_partitioning_on_real_data(
    matcher_mock, dataset_refs, dataset_ground_truth, dataset_fwd_graph
):
    algorithm = HierarchicalAgglomerativeClustering(dataset_refs)

    actual = algorithm(dataset_fwd_graph)

    assert is_partition_over(dataset_refs, actual)
    score = twi(dataset_ground_truth, actual)
    assert 0 <= score <= 1
