import pytest

from matchescu.clustering._mcl import MarkovClustering


@pytest.fixture
def mcl(all_refs):
    return MarkovClustering(all_refs)


def test_single_cluster_on_default_data(mcl, similarity_graph):
    clusters = mcl(similarity_graph)

    assert len(clusters) > 0
