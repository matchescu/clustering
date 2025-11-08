import itertools
from functools import reduce

import pytest

from matchescu.clustering._gacl import ACLClustering
from matchescu.clustering._mcl import MarkovClustering
from matchescu.similarity import ReferenceGraph


@pytest.fixture
def chain_reference_graph(ref, matcher_mock, source):
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
def branched_reference_graph(ref, matcher_mock, source):
  edge_spec = [
      (ref("a", source), ref("b", source)),
      (ref("a", source), ref("c", source)),
      (ref("b", source), ref("d", source)),
      (ref("c", source), ref("d", source)),
  ]
  sim_graph = reduce(
      lambda g, pair: g.add(*pair),
      edge_spec,
      ReferenceGraph(matcher_mock, directed=True),
  )
  return sim_graph


@pytest.fixture
def cyclic_reference_graph(ref, matcher_mock, source):
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
def global_acl(all_refs):
    return ACLClustering(all_refs)


def is_partition_over(all_refs: set, clusters: frozenset[frozenset]) -> tuple[bool, str]:
    if frozenset(all_refs) != frozenset.union(*clusters):
        return False, "clusters do not contain all elements"
    return all(
        (len(a & b) == 0) for a, b in itertools.combinations(clusters, 2)
    ), "clusters are overlapping"


def test_single_cluster_on_default_data(global_acl, reference_graph):
    clusters = global_acl(reference_graph)

    assert len(clusters) == 2  # returns {{a}, {c, b, d}}


def test_global_acl_chain_partition(all_refs, chain_reference_graph):
    algo = ACLClustering(all_refs)
    clusters = algo(chain_reference_graph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg


def test_global_acl_branched_graph(all_refs, branched_reference_graph):
    algo = ACLClustering(all_refs)
    clusters = algo(branched_reference_graph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg


def test_global_acl_cyclic_graph(all_refs, cyclic_reference_graph):
    algo = ACLClustering(all_refs)
    clusters = algo(cyclic_reference_graph)

    ok, msg = is_partition_over(all_refs, clusters)
    assert ok, msg
    assert len(clusters) == 3
