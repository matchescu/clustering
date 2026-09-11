from functools import reduce

import numpy as np
import pytest
import scipy.sparse as sp
from matchescu.similarity import ReferenceGraph
from pyresolvemetrics import (
    adjusted_rand_index,
    cluster_comparison_measure,
    pair_comparison_measure,
    twi,
)

from matchescu.clustering._spectral import SpectralClustering
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


def _dangling_transition_matrix():
    """A row-normalised transition matrix with two dangling rows (zero
    out-degree). Node 0 and node 3 can reach no one; nodes 1 and 2 point at
    each other, so probability mass would leak every iteration unless the
    dangling redistribution is in place."""
    n = 4
    data = np.array([1.0, 1.0])
    row = np.array([1, 2])
    col = np.array([2, 1])
    adjacency = sp.csr_matrix((data, (row, col)), shape=(n, n))
    out_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.divide(
            1.0, out_degrees, out=np.zeros_like(out_degrees), where=out_degrees > 0
        )
    return sp.diags(inv) @ adjacency.tocsr()


def test_power_iter_converges_with_dangling_nodes(all_refs):
    algorithm = SpectralClustering(all_refs, max_power_iterations=1000)
    transition = _dangling_transition_matrix()

    stationary = algorithm._power_iter(transition)

    assert stationary.shape == (4,)
    assert np.isfinite(stationary).all()
    assert pytest.approx(stationary.sum(), abs=1e-9) == 1.0
    assert (stationary >= 0).all()


def test_power_iter_raises_when_dangling_convergence_fails(all_refs):
    """A deliberately tiny iteration budget must still raise rather than
    silently returning a non-converged, non-stochastic vector."""
    algorithm = SpectralClustering(all_refs, max_power_iterations=2)
    transition = _dangling_transition_matrix()

    with pytest.raises(RuntimeError, match="Power iteration did not converge"):
        algorithm._power_iter(transition)


def test_spectral_clustering_handles_dangling_after_square(
    all_refs, source, ref, matcher_mock
):
    """Regression test for graphs whose adjacency square (beta = 2) leaves
    nodes with zero out-degree. The directed chain a->b->c->d produces
    dangling nodes c and d in A^2; clustering must complete without raising
    a power-iteration convergence error."""
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
    ]
    sim_graph = reduce(
        lambda g, pair: g.add(matcher_mock(*pair)),
        edge_spec,
        ReferenceGraph(directed=True),
    )
    algorithm = SpectralClustering(all_refs, max_power_iterations=1000)

    clusters = algorithm(sim_graph)

    assert is_partition_over(all_refs, clusters)
