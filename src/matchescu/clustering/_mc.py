import networkx as nx
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

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        return matrix / matrix.sum(axis=0)

    def _expand(self, matrix: np.ndarray) -> np.ndarray:
        return np.linalg.matrix_power(matrix, self._expansion_power)

    def _inflate(self, matrix: np.ndarray) -> np.ndarray:
        return np.power(matrix, self._inflation_power)

    def _has_converged(self, matrix: np.ndarray, previous_matrix: np.ndarray) -> bool:
        return np.allclose(matrix, previous_matrix, atol=self._convergence_threshold)

    def _extract_clusters(self, matrix: np.ndarray, index_to_item: dict[int, T]) -> frozenset[frozenset[T]]:
        clusters = frozenset(
            frozenset(np.where(matrix[i] > 0)[0])
            for i in range(matrix.shape[0])
            if matrix[i, i] != 0
        )
        result_clusters = set(
            frozenset(index_to_item[idx] for idx in cluster)
            for cluster in clusters
        )
        clustered_items = set(
            item for cluster in clusters for item in cluster
        )
        singletons = self._items - clustered_items
        for singleton in singletons:
            result_clusters.add(frozenset([singleton]))
        return frozenset(result_clusters)

    def _create_transfer_matrix(self, matches: list[tuple[T, T]], item_to_index: dict[T, int]) -> np.ndarray:
        n = len(self._items)
        matrix = np.zeros((n, n))

        g = nx.DiGraph(matches)
        for left, right in matches:
            left_idx, right_idx = item_to_index[left], item_to_index[right]
            out_degree = g.out_degree(left)
            matrix[left_idx, right_idx] = 1 / out_degree if out_degree > 0 else 0

        return matrix

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        filtered_matches = (
            list(filter(lambda pair: self._simg.weight(*pair) >= self._threshold, matches))
            if self._simg is not None else matches
        )
        # create two order preserving indexes so we can use numpy
        item_to_index = {item: idx for idx, item in enumerate(self._items)}
        index_to_item = {idx: item for item, idx in item_to_index.items()}

        # transfer matrix
        transfer_matrix = self._create_transfer_matrix(filtered_matches, item_to_index)
        flow_matrix = transfer_matrix @ transfer_matrix

        # Markov clustering
        matrix = self._add_self_loops(flow_matrix)
        matrix = self._normalize(matrix)

        for _ in range(self._max_iterations):
            previous_matrix = matrix.copy()
            matrix = self._expand(matrix)
            matrix = self._inflate(matrix)
            matrix = self._normalize(matrix)

            if self._has_converged(matrix, previous_matrix):
                break

        # convert to frozenset
        return self._extract_clusters(matrix, index_to_item)
