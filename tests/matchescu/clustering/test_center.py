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
    "matcher_mock,reference_graph,all_refs,min_match_threshold",
    [(TEST_EDGES, list(TEST_EDGES.keys()), TEST_EDGES, min(TEST_EDGES.values()))],
    indirect=True,
)
def test_basic_scenario(
    matcher_mock,
    reference_graph,
    all_refs,
    min_match_threshold,
):
    pc = ParentCenterClustering(all_refs, 0.0)

    clusters = pc(reference_graph)

    assert len(clusters) == 4
    n_clustered_items = len(set(elem for cluster in clusters for elem in cluster))
    assert len(all_refs) == n_clustered_items


TEST_CYCLE_EDGES = {
    ("u", "v"): 0.6,
    ("v", "w"): 0.5,
    ("w", "u"): 0.8,
}


@pytest.mark.parametrize(
    "matcher_mock,reference_graph,all_refs,min_match_threshold",
    [
        (
            TEST_CYCLE_EDGES,
            list(TEST_CYCLE_EDGES.keys()),
            {"u", "v", "w", "x"},
            min(TEST_CYCLE_EDGES.values()),
        )
    ],
    indirect=True,
)
def test_cycle(
    matcher_mock,
    reference_graph,
    all_refs,
    min_match_threshold,
):
    pc = ParentCenterClustering(all_refs, min_match_threshold)

    clusters = pc(reference_graph)
    assert len(clusters) == 2
