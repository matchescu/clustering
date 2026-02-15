from typing import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
from sklearn.cluster import KMeans

from matchescu.clustering._base import (
    ClusteringAlgorithm,
    T,
    NxDirectedMixin,
    SingletonHandlerMixin,
)
from matchescu.similarity import ReferenceGraph


class SpectralClustering(
    ClusteringAlgorithm[T], SingletonHandlerMixin[T], NxDirectedMixin
):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        max_cluster_count: int | None = None,
        alpha: float = 0.85,
        beta: int = 2,
        epsilon: float = 1e-5,
        detect_wcc: bool = True,
        min_component_size: int = 3,
        k_means_random_state: int = 21359482,
        max_power_iterations: int = 1000,
    ):
        if min_component_size < 3:
            raise ValueError("minimum detected weak component size must be at least 3")
        if beta < 1:
            raise ValueError("beta must be a positive integer ≥ 1")

        super().__init__(all_refs, threshold)
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon
        self._k = max_cluster_count
        self._detect_wcc = detect_wcc
        self._min_component_size = min_component_size
        self._kmeans_random = k_means_random_state
        self._max_iter = max_power_iterations

    @classmethod
    def _transition_matrix(cls, adjacency_matrix: sp.csr_matrix):
        out_degrees = np.asarray(adjacency_matrix.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv = np.divide(
                1.0, out_degrees, out=np.zeros_like(out_degrees), where=out_degrees > 0
            )
        D_inv = sp.diags(inv)
        return D_inv @ adjacency_matrix

    def _power_iter(
        self, transition_matrix: sp.csr_matrix, alpha=0.85, epsilon=1e-5
    ) -> np.ndarray:
        n = transition_matrix.shape[0]
        pi = np.ones(n) / n  # uniform initial distribution
        teleport = np.full(n, (1 - alpha) / n)

        for i in range(1, self._max_iter + 1):
            pi_new = alpha * transition_matrix.T @ pi + teleport
            pi_new /= pi_new.sum()
            diff = np.linalg.norm(pi_new - pi)
            if diff < epsilon:
                return pi_new
            pi = pi_new

        raise RuntimeError(
            f"Power iteration did not converge after {self._max_iter} iterations."
        )

    @classmethod
    def _rw_laplacian(
        cls,
        transition_matrix: sp.csr_matrix,  # NxN matrix
        stationary_distribution: np.ndarray,  # 1xN matrix
    ) -> np.ndarray:
        n = transition_matrix.shape[0]
        pi_sqrt = np.sqrt(stationary_distribution)
        pi_inv_sqrt = 1.0 / pi_sqrt
        pi_sqrt = sp.diags(pi_sqrt)
        pi_inv_sqrt = sp.diags(pi_inv_sqrt)

        s = pi_sqrt @ transition_matrix @ pi_inv_sqrt
        random_walk_laplacian = sp.eye(n) - 0.5 * (s + s.T)

        return random_walk_laplacian

    def _spectral_clustering(
        self, adjacency_matrix: sp.csr_matrix, max_clusters: int | None = None
    ):
        M = self._transition_matrix(adjacency_matrix)
        pi = self._power_iter(M, self._alpha)
        L_rw = self._rw_laplacian(M, pi)

        if max_clusters is None or max_clusters < 2:
            max_clusters = max(2, int(adjacency_matrix.shape[0] // 2))

        # smallest k eigenvectors of the random walk Laplacian
        eigenvalues, eigenvectors = sp_linalg.eigsh(L_rw, k=max_clusters, which="SM")
        gaps = np.diff(eigenvalues)

        if max_clusters > 2:
            k = np.argmax(gaps[1:]) + 2
        elif max_clusters == 2:
            k = 2 if gaps[0] > self._epsilon else 1
        else:
            k = 1

        embedding = eigenvectors[:, :k]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embedding /= norms

        labels = KMeans(n_clusters=k, random_state=self._kmeans_random).fit_predict(
            embedding
        )

        return labels, eigenvalues, eigenvectors, embedding, pi, gaps

    def _extract_graph_clusters(
        self, g: nx.Graph, max_clusters: int | None = None
    ) -> Iterable[Iterable[T]]:
        nodes = list(g.nodes)
        adj_matrix = nx.to_scipy_sparse_array(g).power(n=self._beta)

        lbl, lambda_, v, emb, pi, gaps = self._spectral_clustering(
            adj_matrix, max_clusters
        )
        clusters = {}
        for idx, cluster_label in enumerate(lbl):
            clusters.setdefault(cluster_label, []).append(nodes[idx])
        for cluster_no, cluster in clusters.items():
            yield cluster

    def __call__(self, similarity_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = self._to_directed(similarity_graph, self._threshold)
        clusters = []
        wcc = nx.weakly_connected_components(g)
        if self._detect_wcc:
            for c in wcc:
                if len(c) >= self._min_component_size:
                    clusters.extend(self._extract_graph_clusters(g.subgraph(c)))
                else:
                    clusters.append(c)
        else:
            if self._k is None or self._k < 2:
                max_clusters = sum(n // 2 for n in map(len, wcc))
            else:
                max_clusters = self._k
            clusters.extend(self._extract_graph_clusters(g, max_clusters))
        return self._add_singletons(self._items, clusters)
