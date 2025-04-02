import itertools
from typing import TypeVar, Generic

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReferenceIdentifier

T = TypeVar("T", bound=EntityReferenceIdentifier)


class HierarchicalAgglomerativeClustering(Generic[T]):
    def __init__(
        self,
        all_comparisons: BinaryComparisonSpace,
        distance_function: str = "cosine",
        max_cluster_distance: float = 0.0,
    ) -> None:
        self._items = list(set(item for pair in all_comparisons for item in pair))
        self._fcluster_threshold = max_cluster_distance
        self._distance_function = distance_function
        self._linkage_method = "ward"
        self._clustering_criterion = "distance"

    def _distance_matrix(self, similarity_graph: SimilarityGraph) -> np.ndarray:
        g = nx.DiGraph()
        g.add_nodes_from(self._items)
        g.add_weighted_edges_from(
            itertools.starmap(
                lambda u, v, data: (u, v, data.get("weight", 0.0)),
                similarity_graph.edges,
            )
        )
        sim_matrix = nx.to_numpy_array(
            g, nodelist=self._items, weight="weight"
        ) + np.eye(len(self._items))
        return 1 - ((sim_matrix + sim_matrix.T) / 2)

    def __call__(self, similarity_graph: SimilarityGraph) -> frozenset[frozenset[T]]:
        distance_matrix = self._distance_matrix(similarity_graph)

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
