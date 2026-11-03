import pytest
from matchescu.similarity import MatchResult, ReferenceGraph
from pyresolvemetrics import twi

from matchescu.clustering._mcl import MarkovClustering
from tests.testutil import is_partition_over


@pytest.fixture
def mcl(all_refs):
    return MarkovClustering(all_refs)


@pytest.fixture
def nodes_without_edges_digraph(ref, source):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("c", source), ref("d", source)),
    ]
    graph = ReferenceGraph(directed=True)
    for left, right in edge_spec:
        graph.add(MatchResult(left.id, right.id, 0, [1.0, 0.0]))
    return graph


@pytest.fixture
def below_threshold_digraph(ref, source):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
    ]
    graph = ReferenceGraph(directed=True)
    for left, right in edge_spec:
        graph.add(MatchResult(left.id, right.id, 1, [0.75, 0.25]))
    return graph


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


def test_on_graph_without_matches(mcl, all_refs, nodes_without_edges_digraph):
    clusters = mcl(nodes_without_edges_digraph)

    assert is_partition_over(all_refs, clusters)
    assert clusters == frozenset(frozenset({ref_id}) for ref_id in all_refs)


def test_when_all_edges_below_threshold(mcl, all_refs, below_threshold_digraph):
    clusters = mcl(below_threshold_digraph)

    assert is_partition_over(all_refs, clusters)
    assert clusters == frozenset(frozenset({ref_id}) for ref_id in all_refs)


@pytest.mark.parametrize("all_refs", [["a", "b", "c", "d", "e"]], indirect=True)
def test_on_graph_with_isolated_items(mcl, all_refs, chain_digraph, ref_id, source):
    clusters = mcl(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert frozenset({ref_id("e", source)}) in clusters


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
