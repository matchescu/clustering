import itertools
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform  # use condensed-form utility for HAC

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
        # Average linkage natively accepts precomputed distances, unlike Ward.
        self._linkage_method = "average"
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

        # Work with a symmetrized similarity matrix, then invert to get distances that
        # HAC can consume directly; enforce zero diagonals for squareform.
        sim_matrix = (sim_matrix + sim_matrix.T) / 2
        distance_matrix = 1 - sim_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        distance_matrix = self._distance_matrix(reference_graph)

        # squareform converts the symmetric matrix to the condensed vector
        # expected by SciPy linkage when supplying distances directly.
        condensed_distance_matrix = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distance_matrix, method=self._linkage_method)

        cluster_assignments = fcluster(
            Z, self._fcluster_threshold, criterion=self._clustering_criterion
        )

        unique_clusters = np.unique(cluster_assignments)
        return frozenset(
            frozenset(self._items[idx] for idx in np.where(cluster_assignments == c)[0])
            for c in unique_clusters
        )
