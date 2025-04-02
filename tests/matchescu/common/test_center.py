import pytest

from matchescu.clustering._center import ParentCenterClustering

TEST_EDGES = {
    ("A", "B"): 0.8,
    ("C", "B"): 0.9,
    ("B", "D"): 0.7,
    ("E", "D"): 0.6,
    ("D", "F"): 0.8,
    ("G", "F"): 0.9,
    ("H", "G"): 0.7,
    ("I", "H"): 0.6,
    ("J", "I"): 0.8,
    ("K", "J"): 0.9,
    ("L", "K"): 0.7,
    ("M", "L"): 0.6,
    ("N", "M"): 0.8,
    ("O", "N"): 0.9,
    ("P", "O"): 0.7,
    ("Q", "P"): 0.6,
    ("R", "Q"): 0.8,
    ("S", "R"): 0.9,
}


@pytest.mark.parametrize(
    "comparison_space,matcher_mock,similarity_graph,min_match_threshold",
    [(TEST_EDGES, TEST_EDGES, list(TEST_EDGES.keys()), min(TEST_EDGES.values()))],
    indirect=True,
)
def test_basic_scenario(
    comparison_space, matcher_mock, similarity_graph, min_match_threshold
):
    make_clusters = ParentCenterClustering(comparison_space, 0.0)

    clusters = make_clusters(similarity_graph)

    assert len(clusters) == 4
    n_items = len(set(item for pair in comparison_space for item in pair))
    n_clustered_items = len(set(elem for cluster in clusters for elem in cluster))
    assert n_items == n_clustered_items
