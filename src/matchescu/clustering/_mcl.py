from collections.abc import Iterable

import markov_clustering as mc
import networkx as nx

from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import T, ClusteringAlgorithm


class MarkovClustering(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        inflation_power: float = 2.0,
        expansion_power: int = 2,
        prune_threshold: float = 0.001,
    ):
        super().__init__(all_refs, threshold)
        self._item_to_index = {item: idx for idx, item in enumerate(self._items)}
        self._expansion_power = expansion_power
        self._inflation_power = inflation_power
        self._prune_threshold = prune_threshold

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = nx.DiGraph()

        for u, v in reference_graph.matches(self._threshold):
            g.add_edge(u, v, weight=reference_graph.weight(u, v))

        linked_ref_ids = list(g.nodes)
        adj_matrix = nx.to_numpy_array(g, weight="weight")
        adj_matrix[adj_matrix < 0] = 0

        result = mc.run_mcl(
            adj_matrix,
            self._expansion_power,
            self._inflation_power,
            pruning_threshold=self._prune_threshold,
        )
        cluster_indices = mc.get_clusters(result)

        singletons = set(self._items) - set(linked_ref_ids)
        clusters = set()
        for c in cluster_indices:
            cluster = frozenset(linked_ref_ids[idx] for idx in c)
            singletons.difference_update(cluster)
            clusters.add(cluster)
        for singleton in singletons:
            clusters.add(frozenset([singleton]))

        return frozenset(clusters)
