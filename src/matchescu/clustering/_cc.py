import networkx as nx

from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReference


class ConnectedComponents:
    def __init__(self, threshold: float | None) -> None:
        self.__threshold = threshold

    def __call__(self, similarity_graph: SimilarityGraph) -> frozenset[frozenset[EntityReference]]:
        g = nx.Graph()
        for node in similarity_graph.nodes:
            g.add_node(node)
        for u, v, data in similarity_graph.matches():
            if self.__threshold is not None and data.get("weight", 0) < self.__threshold:
                continue
            g.add_edge(u, v)
        clusters = nx.connected_components(g)
        result = frozenset(
            frozenset(node for node in cluster)
            for cluster in clusters
        )
        return result