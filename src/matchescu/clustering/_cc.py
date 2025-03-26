import networkx as nx

from matchescu.similarity import SimilarityGraph, MatchEdgeType
from matchescu.typing import EntityReferenceIdentifier


class ConnectedComponents:
    def __init__(self, threshold: float | None) -> None:
        self.__threshold = threshold

    def __call__(self, similarity_graph: SimilarityGraph) -> frozenset[frozenset[EntityReferenceIdentifier]]:
        g = nx.Graph()
        for node in similarity_graph.nodes:
            g.add_node(node)
        for u, v, data in similarity_graph.edges:
            if data.get("type") != MatchEdgeType.MATCH:
                continue
            if self.__threshold is not None and data.get("weight", 0) < self.__threshold:
                continue
            g.add_edge(u.id, v.id)
        clusters = nx.connected_components(g)
        result = frozenset(
            frozenset(node for node in cluster)
            for cluster in clusters
        )
        return result