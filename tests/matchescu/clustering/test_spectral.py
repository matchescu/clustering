import pytest

from matchescu.clustering._spectral import SpectralClustering
from pyresolvemetrics import (
    pair_comparison_measure,
    cluster_comparison_measure,
    adjusted_rand_index,
    twi,
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


@pytest.mark.skip(reason="only run this locally - not in CI")
@pytest.mark.parametrize(
    "dataset",
    [
        "abt-buy",
        "amazon-google",
        "beer",
        "dblp-scholar",
    ],
    indirect=True,
)
def test_partitioning_on_real_data(
    benchmark, matcher_mock, dataset_refs, dataset_ground_truth, dataset_bidi_graph
):
    algorithm = SpectralClustering(dataset_refs, threshold=0.4, detect_wcc=True)

    actual = benchmark(algorithm, dataset_bidi_graph)

    assert is_partition_over(dataset_refs, actual)
    metrics = [
        pair_comparison_measure,
        cluster_comparison_measure,
        adjusted_rand_index,
        twi,
    ]
    scores = [metric(dataset_ground_truth, actual) for metric in metrics]
    for score in scores:
        assert score > 0.5
        assert score <= 1
