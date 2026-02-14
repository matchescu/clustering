import pytest

from matchescu.clustering._leiden import LeidenPartitioning
from pyresolvemetrics import twi
from tests.testutil import is_partition_over


@pytest.fixture
def leiden(all_refs):
    return LeidenPartitioning(all_refs)


def test_on_chain(leiden, all_refs, chain_digraph):
    clusters = leiden(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2, "expected chain to be broken up into singletons"


def test_on_ring(leiden, all_refs, ring_digraph):
    clusters = leiden(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2


def test_clique(leiden, all_refs, clique_digraph):
    clusters = leiden(clique_digraph)

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
def test_ring_with_cliques(leiden, all_refs, ring_with_cliques_digraph):
    clusters = leiden(ring_with_cliques_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}: {clusters}"
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [3, 3, 4], f"Expected cluster sizes [3, 3, 4], got {sizes}"


@pytest.mark.skip(reason="only run this locally - not in CI")
def test_partitioning_on_real_data(
    matcher_mock, dataset_refs, dataset_ground_truth, dataset_fwd_graph
):
    algorithm = LeidenPartitioning(dataset_refs, threshold=0.4)

    actual = algorithm(dataset_fwd_graph)

    assert is_partition_over(dataset_refs, actual)
    score = twi(dataset_ground_truth, actual)
    assert 0 <= score <= 1
