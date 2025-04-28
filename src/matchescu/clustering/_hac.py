import itertools
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import ClusteringAlgorithm, T


class HierarchicalAgglomerativeClustering(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        distance_function: str = "cosine",
        max_cluster_distance: float = 1.0,
    ) -> None:
        super().__init__(all_refs, 0.0)
        self._fcluster_threshold = max_cluster_distance
        self._distance_function = distance_function
        self._linkage_method = "ward"
        self._clustering_criterion = "distance"

    def _distance_matrix(self, reference_graph: ReferenceGraph) -> np.ndarray:
        g = nx.DiGraph()
        g.add_nodes_from(self._items)
        g.add_weighted_edges_from(
            itertools.starmap(
                lambda u, v, data: (u, v, data.get("weight", 0.0)),
                reference_graph.edges,
            )
        )
        sim_matrix = nx.to_numpy_array(
            g, nodelist=self._items, weight="weight"
        ) + np.eye(len(self._items))
        return 1 - ((sim_matrix + sim_matrix.T) / 2)

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        distance_matrix = self._distance_matrix(reference_graph)

        # compute hierarchical clusters based on average
        condensed_distance_matrix = pdist(distance_matrix, self._distance_function)
        Z = linkage(condensed_distance_matrix, method=self._linkage_method)

        # flatten the clusters based on distance
        cluster_assignments = fcluster(
            Z, self._fcluster_threshold, criterion=self._clustering_criterion
        )

        # map cluster assignments back to items
        unique_clusters = np.unique(cluster_assignments)
        return frozenset(
            frozenset(self._items[idx] for idx in np.where(cluster_assignments == c)[0])
            for c in unique_clusters
        )
