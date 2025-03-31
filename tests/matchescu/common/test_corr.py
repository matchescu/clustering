import pytest

from matchescu.clustering._corr import WeightedCorrelationClustering


@pytest.fixture
def corr(comparison_space, min_match_threshold):
    return WeightedCorrelationClustering(comparison_space, min_match_threshold)


def test_single_cluster_on_default_data(corr, similarity_graph):
    clusters = corr(similarity_graph)

    assert len(clusters) > 0
