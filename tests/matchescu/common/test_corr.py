import pytest

from matchescu.clustering._corr import WeightedCorrelationClustering


@pytest.fixture
def corr(comparison_space):
    return WeightedCorrelationClustering(comparison_space)


def test_single_cluster_on_default_data(corr, comparison_space):
    clusters = corr(comparison_space)

    assert len(clusters) > 0
