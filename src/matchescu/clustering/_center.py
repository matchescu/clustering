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
        path = []

        while parents[node] != node:
            path.append(node)
            node = parents[node]

        for n in path:
            # path compression using the last node in the path
            parents[n] = node

        return node

    def _construct_dag(self, reference_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(self._items)
        seen_pairs = set()
        for u, v in reference_graph.matches(self._threshold):
            if (v, u) in seen_pairs:
                continue
            w = max(
                reference_graph.weight(u, v),
                reference_graph.weight(v, u),
            )
            graph.add_edge(u, v, weight=w)
            seen_pairs.add((u, v))
        return graph

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        graph = self._construct_dag(reference_graph)
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
