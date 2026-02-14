import pytest

from matchescu.clustering._spectral import SpectralClustering
from pyresolvemetrics import (
    twi,
    cluster_comparison_measure,
    pair_comparison_measure,
    adjusted_rand_index,
)
from tests.testutil import is_partition_over


@pytest.fixture
def spectral(all_refs):
    return SpectralClustering(all_refs)


def test_on_chain(spectral, all_refs, chain_digraph):
    clusters = spectral(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2, "expected chain to be broken"


def test_on_ring(spectral, all_refs, ring_digraph):
    clusters = spectral(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2, "expected ring to be broken"


def test_clique(spectral, all_refs, clique_digraph):
    clusters = spectral(clique_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2, "Clique should be divided"


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
def test_ring_with_cliques(spectral, all_refs, ring_with_cliques_digraph):
    clusters = spectral(ring_with_cliques_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 3, "Clean separation along bridges expected"


# @pytest.mark.skip(reason="only run this locally - not in CI")
def test_partitioning_on_real_data(
    matcher_mock, dataset_refs, dataset_ground_truth, dataset_fwd_graph
):
    algorithm = SpectralClustering(dataset_refs, threshold=0.4)

    actual = algorithm(dataset_fwd_graph)

    assert is_partition_over(dataset_refs, actual)
    metrics = [
        pair_comparison_measure,
        cluster_comparison_measure,
        adjusted_rand_index,
        twi,
    ]
    scores = [metric(dataset_ground_truth, actual) for metric in metrics]
    assert all(0.95 <= score <= 1 for score in scores)
