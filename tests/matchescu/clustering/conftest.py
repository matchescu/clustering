from functools import reduce

import pytest

from matchescu.similarity import ReferenceGraph


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
