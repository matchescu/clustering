import numpy as np
from typing import Generic, TypeVar

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph


T = TypeVar("T")


class MarkovClustering(Generic[T]):
    def __init__(
        self,
        comparison_space: BinaryComparisonSpace,
        similarity_graph: SimilarityGraph | None = None,
        expansion_power: int = 2,
        inflation_power: int = 2,
        loop_value: int = 1,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        min_edge_weight: float = 0.5,
    ):
        self._items = set(ref_id for pair in comparison_space for ref_id in pair)
        self._simg = similarity_graph
        self._expansion_power = expansion_power
        self._inflation_power = inflation_power
        self._loop_value = loop_value
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold
        self._threshold = min_edge_weight

    def _add_self_loops(self, matrix: np.ndarray) -> np.ndarray:
        np.fill_diagonal(matrix, self._loop_value)
        return matrix

    @staticmethod
    def _inflate(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix**2
        return matrix / matrix.sum(axis=0)

    def _has_converged(self, matrix: np.ndarray, previous_matrix: np.ndarray) -> bool:
        return np.allclose(matrix, previous_matrix, atol=self._convergence_threshold)

    def _extract_clusters(
        self, matrix: np.ndarray, index_to_item: dict[int, T]
    ) -> frozenset[frozenset[T]]:
        clusters = []
        for i in range(matrix.shape[0]):
            if matrix[i, i] != 0:
                cluster = set(np.where(matrix[i] > 0)[0])
                if cluster not in clusters:
                    clusters.append(cluster)
        clustered_items = set()
        result_clusters = []
        for cluster in clusters:
            cluster_items = frozenset(index_to_item[idx] for idx in cluster)
            result_clusters.append(cluster_items)
            clustered_items.update(cluster_items)
        # Add singletons
        singletons = self._items - clustered_items
        for singleton in singletons:
            result_clusters.append(frozenset([singleton]))
        return frozenset(result_clusters)

    def _create_transition_matrix(
        self, matches: list[tuple[T, T]], item_to_index: dict[T, int]
    ) -> np.ndarray:
        n = len(self._items)
        adj_matrix = np.zeros((n, n))

        # g = nx.DiGraph(matches)

        for left, right in matches:
            left_idx, right_idx = item_to_index[left], item_to_index[right]
            adj_matrix[left_idx, right_idx] = 1
        adj_matrix = self._add_self_loops(adj_matrix)

        return adj_matrix / adj_matrix.sum(axis=0)

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        filtered_matches = list(
            filter(lambda pair: self._simg.weight(*pair) >= self._threshold, matches)
            if self._simg is not None
            else matches
        )
        # create two order preserving indexes so we can use numpy
        item_to_index = {item: idx for idx, item in enumerate(self._items)}
        index_to_item = {idx: item for item, idx in item_to_index.items()}

        # transfer matrix
        transition_matrix = self._create_transition_matrix(
            filtered_matches, item_to_index
        )
        flow_matrix = self._inflate(transition_matrix @ transition_matrix)

        # Markov clustering
        for _ in range(self._max_iterations):
            previous_matrix = flow_matrix.copy()
            flow_matrix = self._inflate(flow_matrix @ flow_matrix)

            if self._has_converged(flow_matrix, previous_matrix):
                break

        # convert to frozenset
        return self._extract_clusters(flow_matrix, index_to_item)
