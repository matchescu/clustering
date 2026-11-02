from matchescu.similarity import MatchResult, ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier as RefId

from matchescu.clustering import (
    LeidenPartitioning,
    SpectralClustering,
    WeaklyConnectedComponents,
)


def _build_graph(edges):
    graph = ReferenceGraph(directed=True)
    for left, right, weight in edges:
        graph.add(MatchResult(left, right, 1, [1.0 - weight, weight]))
    return graph


def _two_cliques_with_bridges(size=12, bridges=2):
    refs = [RefId(i, "s") for i in range(size)]
    half = size // 2
    edges = []
    for start in (0, half):
        for left in range(start, start + half):
            for right in range(left + 1, start + half):
                edges.append((refs[left], refs[right], 1.0))
    for index in range(bridges):
        edges.append((refs[index], refs[half + index], 1.0))
    return refs, edges


def test_clustering_items_preserve_insertion_order_and_deduplicate():
    refs = [RefId(3, "s"), RefId(1, "s"), RefId(3, "s"), RefId(2, "s")]

    algorithm = WeaklyConnectedComponents(refs)

    assert algorithm._items == [RefId(3, "s"), RefId(1, "s"), RefId(2, "s")]


def test_leiden_is_reproducible_with_same_seed():
    refs, edges = _two_cliques_with_bridges()
    first = LeidenPartitioning(refs, threshold=0.5, seed=7)(_build_graph(edges))
    second = LeidenPartitioning(refs, threshold=0.5, seed=7)(_build_graph(edges))

    assert first == second


def test_leiden_is_reproducible_with_default_seed():
    refs, edges = _two_cliques_with_bridges()
    first = LeidenPartitioning(refs, threshold=0.5)(_build_graph(edges))
    second = LeidenPartitioning(refs, threshold=0.5)(_build_graph(edges))

    assert first == second


def test_spectral_is_reproducible():
    refs, edges = _two_cliques_with_bridges()
    first = SpectralClustering(refs, threshold=0.5)(_build_graph(edges))
    second = SpectralClustering(refs, threshold=0.5)(_build_graph(edges))

    assert first == second
