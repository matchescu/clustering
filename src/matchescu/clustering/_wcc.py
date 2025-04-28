from collections.abc import Iterable

import networkx as nx

from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReferenceIdentifier

from matchescu.clustering._base import T, ClusteringAlgorithm


class WeaklyConnectedComponents(ClusteringAlgorithm[T]):
    def __init__(self, all_refs: Iterable[T], threshold: float | None) -> None:
        super().__init__(all_refs, threshold)

    def __call__(
        self, similarity_graph: SimilarityGraph
    ) -> frozenset[frozenset[EntityReferenceIdentifier]]:
        g = nx.DiGraph()
        g.add_nodes_from(self._items)
        for u, v in similarity_graph.matches():
            g.add_edge(u, v, weight=similarity_graph.weight(u, v))
        return frozenset(
            frozenset(v for v in comp) for comp in nx.weakly_connected_components(g)
        )
