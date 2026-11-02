from collections.abc import Iterable

import igraph as ig
import leidenalg as la
from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import ClusteringAlgorithm, T


class LeidenPartitioning(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        resolution: float = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(all_refs, threshold)
        self._resolution = resolution
        self._seed = seed

    def _to_igraph(self, reference_graph: ReferenceGraph) -> ig.Graph:
        idx = {n: i for i, n in enumerate(self._items)}
        edges = [(u, v) for u, v in reference_graph.matches(self._threshold)]
        graph = ig.Graph(
            n=len(self._items),
            edges=[(idx[u], idx[v]) for u, v in edges],
            edge_attrs={"weight": [reference_graph.weight(u, v) for u, v in edges]},
            directed=reference_graph.directed,
        )
        return graph

    def __call__(self, similarity_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        graph = self._to_igraph(similarity_graph)
        result = la.find_partition(graph, la.ModularityVertexPartition, seed=self._seed)
        return frozenset(
            frozenset(self._items[node] for node in cluster) for cluster in result
        )
