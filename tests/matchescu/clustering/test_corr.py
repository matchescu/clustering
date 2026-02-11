import pytest

from matchescu.clustering._corr import WeightedCorrelationClustering


@pytest.fixture
def corr(all_refs, min_match_threshold):
    return WeightedCorrelationClustering(all_refs, min_match_threshold)


def test_single_cluster_on_default_data(corr, reference_graph):
    clusters = corr(reference_graph)

    assert len(clusters) > 0
