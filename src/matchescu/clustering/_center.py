from collections import defaultdict
from collections.abc import Iterable

import networkx as nx
from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import T, ClusteringAlgorithm


class ParentCenterClustering(ClusteringAlgorithm[T]):
    def __init__(self, all_refs: Iterable[T], threshold: float = 0.75) -> None:
        super().__init__(all_refs, threshold)

    @staticmethod
    def _find_root(parents, node):
        """
        Finds the root parent of a node in the parent dictionary.
        """
        if parents[node] == node:
            return node

        return ParentCenterClustering._find_root(parents, parents[node])

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        graph = nx.DiGraph()
        graph.add_nodes_from(self._items)
        graph.add_edges_from(reference_graph.matches(self._threshold))

        parents = {node: node for node in self._items}
        updated = True

        while updated:
            updated = False
            new_parents = parents.copy()
            for node in self._items:
                best_parent = node
                max_similarity = -1

                for predecessor in graph.predecessors(node):
                    similarity = reference_graph.weight(predecessor, node)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_parent = predecessor

                # Update parent to the parent of the best predecessor
                if best_parent != node:
                    root_parent = self._find_root(parents, best_parent)
                    if new_parents[node] != root_parent:
                        new_parents[node] = root_parent
                        updated = True

            parents = new_parents

        result = defaultdict(list)
        for u, v in parents.items():
            result[v].append(u)

        return frozenset(
            frozenset(node for node in cluster) for cluster in result.values()
        )
