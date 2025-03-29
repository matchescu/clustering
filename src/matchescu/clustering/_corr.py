from abc import ABCMeta, abstractmethod
from typing import Generic, Hashable, TypeVar, Set, FrozenSet, List, Tuple
import networkx as nx
import random

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReferenceIdentifier

T = TypeVar("T", bound=EntityReferenceIdentifier)


class WeightedCorrelationClustering(Generic[T]):
    def __init__(self, items: BinaryComparisonSpace, sim_graph: SimilarityGraph|None = None, threshold: float = 0.5) -> None:
        self._items = set(item for pair in items for item in pair)
        self._sim_graph = sim_graph
        self._threshold = threshold

    def _compute_clusters(self, g: nx.DiGraph) -> frozenset[frozenset[T]]:
        if self._sim_graph is None:
            intersection_graph = g.copy()
        else:
            intersection_graph = nx.DiGraph()
            for u, v in g.edges():
                w = self._sim_graph.weight(u, v)
                if w == 0:
                    continue
                intersection_graph.add_edge(u, v, weight=w)

        unvisited_nodes = set(intersection_graph.nodes())
        singletons = self._items - unvisited_nodes
        clusters = [frozenset([singleton]) for singleton in singletons]

        while len(unvisited_nodes) > 0:
            pivot = random.choice(list(unvisited_nodes))
            current_cluster = {pivot}
            for neighbor in intersection_graph.neighbors(pivot):
                if neighbor not in unvisited_nodes:
                    continue
                weight = 1.0 if intersection_graph.has_edge(neighbor, pivot) else 0.0
                if self._sim_graph is not None:
                    weight = self._sim_graph.weight(pivot, neighbor)
                if weight >= self._threshold:
                    current_cluster.add(neighbor)
            clusters.append(frozenset(current_cluster))
            unvisited_nodes.difference_update(current_cluster)
        return frozenset(clusters)

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        g = nx.DiGraph()
        g.add_nodes_from(self._items)
        g.add_edges_from(matches)
        return self._compute_clusters(g)

