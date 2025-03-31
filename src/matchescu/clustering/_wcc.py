import networkx as nx

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReferenceIdentifier


class WeaklyConnectedComponents:
    def __init__(
        self, all_comparisons: BinaryComparisonSpace, threshold: float | None
    ) -> None:
        self._threshold = threshold
        self._items = list(set(item for pair in all_comparisons for item in pair))

    def __call__(
        self, similarity_graph: SimilarityGraph
    ) -> frozenset[frozenset[EntityReferenceIdentifier]]:
        g = nx.DiGraph()
        g.add_nodes_from(similarity_graph.nodes)
        g.add_edges_from(
            edge
            for edge in similarity_graph.matches()
            if similarity_graph.weight(*edge) >= self._threshold
        )
        return frozenset(
            frozenset(v for v in comp) for comp in nx.weakly_connected_components(g)
        )
