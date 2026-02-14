import pytest

from matchescu.clustering._gacl import ACLClustering, SeedStrategy, PartitionStrategy
from pyresolvemetrics import pair_comparison_measure
from tests.testutil import is_partition_over


@pytest.fixture
def global_acl(all_refs):
    return ACLClustering(all_refs)


def test_global_acl_chain_partition(all_refs, chain_digraph):
    algo = ACLClustering(all_refs)
    clusters = algo(chain_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) >= 2, "expected chain to be broken up into singletons"


def test_global_acl_identifies_rings_with_scc(all_refs, ring_digraph):
    algo = ACLClustering(all_refs)
    clusters = algo(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 1


@pytest.mark.parametrize("seed_strategy", [SeedStrategy.DEGREE, SeedStrategy.PAGERANK])
def test_global_acl_no_ring_identification_without_augmentation(
    all_refs, ring_digraph, seed_strategy
):
    algo = ACLClustering(
        all_refs,
        partition_strategy=PartitionStrategy.PAGERANK,
        detect_scc=False,
        seed_strategy=seed_strategy,
    )
    clusters = algo(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 3


@pytest.mark.parametrize("seed_strategy", [SeedStrategy.DEGREE, SeedStrategy.PAGERANK])
def test_global_acl_ring_identification_via_bridge_partitioning(
    all_refs, ring_digraph, seed_strategy
):
    algo = ACLClustering(
        all_refs,
        detect_scc=False,
        seed_strategy=seed_strategy,
    )
    clusters = algo(ring_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 3


def test_global_acl_identifies_cliques(all_refs, clique_digraph):
    algo = ACLClustering(all_refs)
    clusters = algo(clique_digraph)

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
def test_global_acl_ring_with_cliques(global_acl, all_refs, ring_with_cliques_digraph):
    clusters = global_acl(ring_with_cliques_digraph)

    assert is_partition_over(all_refs, clusters)
    assert len(clusters) == 2, f"Expected 3 clusters, got {len(clusters)}: {clusters}"
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [3, 7], f"Expected cluster sizes [3, 7], got {sizes}"


@pytest.mark.skip(reason="only run this locally - not in CI")
def test_global_acl_on_real_data(
    matcher_mock, dataset_refs, dataset_ground_truth, dataset_bidi_graph
):
    algorithm = ACLClustering(
        dataset_refs,
        threshold=0.4,
        partition_strategy=PartitionStrategy.DIRECTED,
        seed_strategy=SeedStrategy.BETWEENNESS_SAMPLED,
        detect_scc=True,
    )

    actual = algorithm(dataset_bidi_graph)

    assert is_partition_over(dataset_refs, actual)
    score = pair_comparison_measure(dataset_ground_truth, actual)
    assert 0.5 <= score <= 1
