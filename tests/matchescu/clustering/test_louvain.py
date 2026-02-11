import itertools
from functools import reduce, partial

import pytest

from matchescu.clustering._louvain import LouvainPartitioning
from matchescu.similarity import ReferenceGraph
from matchescu.typing import EntityReference
from pyresolvemetrics import twi


@pytest.fixture
def chain_digraph(ref, matcher_mock, source):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
    ]
    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        edge_spec,
        ReferenceGraph(matcher_mock, directed=True),
    )
    return sim_graph


@pytest.fixture
def branched_digraph(ref, matcher_mock, source):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("d", source)),
        (ref("a", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
    ]
    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        edge_spec,
        ReferenceGraph(matcher_mock, directed=True),
    )
    return sim_graph


@pytest.fixture
def ring_digraph(ref, matcher_mock, source):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
        (ref("d", source), ref("a", source)),
    ]
    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        edge_spec,
        ReferenceGraph(matcher_mock, directed=True),
    )
    return sim_graph


@pytest.fixture
def clique_digraph(ref, matcher_mock, source):
    a = ref("a", source)
    b = ref("b", source)
    c = ref("c", source)
    d = ref("d", source)
    # All directed pairs: 4 * 3 = 12 edges
    edge_spec = [
        (a, b),
        (a, c),
        (a, d),
        (b, a),
        (b, c),
        (b, d),
        (c, a),
        (c, b),
        (c, d),
        (d, a),
        (d, b),
        (d, c),
    ]
    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        edge_spec,
        ReferenceGraph(matcher_mock, directed=True),
    )
    return sim_graph


@pytest.fixture
def ring_with_cliques_digraph(ref, matcher_mock, source):
    """
    Complex graph structure:
    - Ring: a→b→c→d→a (4 nodes, strongly connected)
    - Clique 1: {e, f, g} fully connected (3 nodes)
    - Clique 2: {h, i, j} fully connected (3 nodes)
    - Bridge edges: a→e, c→h (one-way, so SCCs remain separate)
    """
    a = ref("a", source)
    b = ref("b", source)
    c = ref("c", source)
    d = ref("d", source)
    e = ref("e", source)
    f = ref("f", source)
    g = ref("g", source)
    h = ref("h", source)
    i = ref("i", source)
    j = ref("j", source)

    # Ring edges
    ring_edges = [(a, b), (b, c), (c, d), (d, a)]

    # Clique 1: {e, f, g} - 6 edges
    clique1_edges = [
        (e, f),
        (e, g),
        (f, e),
        (f, g),
        (g, e),
        (g, f),
    ]

    # Clique 2: {h, i, j} - 6 edges
    clique2_edges = [
        (h, i),
        (h, j),
        (i, h),
        (i, j),
        (j, h),
        (j, i),
    ]

    # Bridge edges (one-way: ring → cliques)
    bridge_edges = [(a, e), (c, h)]

    all_edges = ring_edges + clique1_edges + clique2_edges + bridge_edges

    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        all_edges,
        ReferenceGraph(matcher_mock, directed=True),
    )
    return sim_graph


@pytest.fixture
def louvain(all_refs):
    return LouvainPartitioning(all_refs)


def is_partition_over(
    all_refs: set, clusters: frozenset[frozenset]
) -> tuple[bool, str]:
    if frozenset(all_refs) != frozenset.union(*clusters):
        return False, "clusters do not contain all elements"
    is_ok = all((len(a & b) == 0) for a, b in itertools.combinations(clusters, 2))
    return (
        is_ok,
        "valid partition" if is_ok else "overlapping clusters",
    )


def test_louvain_on_chain(louvain, all_refs, chain_digraph):
    clusters = louvain(chain_digraph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg
    assert len(clusters) == 2, "expected chain to be broken up into singletons"


def test_louvain_on_ring(louvain, all_refs, ring_digraph):
    clusters = louvain(ring_digraph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg
    assert len(clusters) == 2


def test_louvain_clique(louvain, all_refs, clique_digraph):
    clusters = louvain(clique_digraph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg
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
def test_louvain_ring_with_cliques(louvain, all_refs, ring_with_cliques_digraph):
    clusters = louvain(ring_with_cliques_digraph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg

    assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}: {clusters}"
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [3, 3, 4], f"Expected cluster sizes [3, 3, 4], got {sizes}"


@pytest.mark.skip(reason="only run this locally - not in CI")
def test_louvain_partitioning_on_real_data(
    matcher_mock, file_digraph, csv_all_refs, csv_ground_truth
):
    reference_graph = reduce(
        lambda g, pair: g.add(*map(partial(EntityReference, value=[]), pair)),
        file_digraph.edges(),
        ReferenceGraph(matcher_mock, directed=True),
    )
    louvain = LouvainPartitioning(csv_all_refs, threshold=0.4)

    actual = louvain(reference_graph)

    ok, msg = is_partition_over(csv_all_refs, actual)
    assert ok, msg
    score = twi(csv_ground_truth, actual)
    assert 0 <= score <= 1
