from collections.abc import Iterable

import networkx as nx

from matchescu.similarity import ReferenceGraph
from matchescu.typing import EntityReferenceIdentifier

from matchescu.clustering._base import T, ClusteringAlgorithm


class ConnectedComponents(ClusteringAlgorithm[T]):
    def __init__(self, all_refs: Iterable[T], threshold: float = 0.75) -> None:
        super().__init__(all_refs, threshold)

    def __call__(
        self, reference_graph: ReferenceGraph
    ) -> frozenset[frozenset[EntityReferenceIdentifier]]:
        if reference_graph.directed:
            raise ValueError(
                "Connected components cannot be computed on directed graphs"
            )

        g = nx.Graph()
        g.add_nodes_from(self._items)
        for u, v in reference_graph.matches(self._threshold):
            g.add_edge(u, v, weight=reference_graph.weight(u, v))
        return frozenset(
            frozenset(v for v in comp) for comp in nx.connected_components(g)
        )
