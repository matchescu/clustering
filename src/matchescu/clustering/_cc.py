from typing import TypeVar, Generic

import networkx as nx

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph


T = TypeVar("T")


class ConnectedComponents(Generic[T]):
    def __init__(
        self,
        comparison_space: BinaryComparisonSpace,
        similarity_graph: SimilarityGraph | None = None,
        threshold: float | None = None,
    ) -> None:
        self.__nodes = {ref_id for pair in comparison_space for ref_id in pair}
        self.__threshold = threshold
        self.__similarity_graph = similarity_graph

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        g = nx.Graph()
        for node in self.__nodes:
            g.add_node(node)
        for u, v in matches:
            if self.__similarity_graph is None:
                g.add_edge(u, v)
                continue
            if not self.__similarity_graph.is_match(u, v):
                continue
            weight = self.__similarity_graph.weight(u, v)
            if self.__threshold is not None and weight < self.__threshold:
                continue
            g.add_edge(u, v)
        clusters = nx.connected_components(g)
        result = frozenset(
            frozenset(node for node in cluster)
            for cluster in clusters
        )
        return result