from collections.abc import Iterable

import networkx as nx
import networkx.algorithms.community.louvain as nx_louvain
from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import T, ClusteringAlgorithm


class LouvainPartitioning(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        louvain_resolution: float = 1,
        louvain_threshold: float = 0.0000001,
    ) -> None:
        super().__init__(all_refs, threshold)
        self._alg_resolution = louvain_resolution
        self._alg_threshold = louvain_threshold

    def _to_digraph(self, reference_graph: ReferenceGraph) -> nx.DiGraph:
        g = nx.DiGraph()
        for u, v in reference_graph.matches(self._threshold):
            g.add_edge(u, v, weight=reference_graph.weight(u, v))
        return g

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        graph = self._to_digraph(reference_graph)

        best = None
        for partition in nx_louvain.louvain_partitions(
            graph,
            weight="weight",
            threshold=self._alg_threshold,
            resolution=self._alg_resolution,
        ):
            best = partition

        unassigned = self._items
        partition = []
        if best is not None:
            assigned = set(node for cluster in best for node in cluster)
            unassigned = set(self._items) - assigned
            partition = [frozenset(node for node in cluster) for cluster in best]
        partition.extend(frozenset([ref]) for ref in unassigned)
        return frozenset(cluster for cluster in partition)
