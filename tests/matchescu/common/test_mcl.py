import pytest

from matchescu.clustering._mc import MarkovClustering


@pytest.fixture
def mcl(comparison_space):
    return MarkovClustering(comparison_space)


def test_single_cluster_on_default_data(mcl, comparison_space):
    clusters = mcl(comparison_space)

    assert len(clusters) > 0
