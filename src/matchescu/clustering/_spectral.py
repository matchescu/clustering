from typing import Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
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
        cluster_count: int | None = None,
        alpha: float = 0.85,
        epsilon: float = 1e-5,
    ):
        super().__init__(all_refs, threshold)
        self._alpha = alpha
        self._epsilon = epsilon
        self._k = cluster_count

    @classmethod
    def _transition_matrix(cls, adjacency_matrix: sp.csr_matrix):
        out_degrees = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
        D_inv = sp.diags(1.0 / out_degrees)
        return D_inv @ adjacency_matrix

    @classmethod
    def _power_iter(
        cls, transition_matrix: sp.csr_matrix, alpha=0.85, epsilon=1e-5, max_iter=200
    ) -> sp.csr_matrix:
        n = transition_matrix.shape[0]
        pi = np.ones(n) / n  # uniform initial distribution

        for i in range(1, max_iter + 1):
            pi_new = alpha * transition_matrix @ pi + ((1 - alpha) / n) * np.ones(n)

            diff = np.linalg.norm(pi_new - pi)
            if diff < epsilon:
                return pi_new
            pi = pi_new

        raise StopIteration(
            f"Power iteration did not converge after {max_iter} iterations."
        )

    @classmethod
    def _rw_laplacian(
        cls,
        transition_matrix: sp.csr_matrix,  # NxN matrix
        stationary_distribution: sp.csr_matrix,  # 1xN matrix
    ) -> sp.csr_matrix:
        n = transition_matrix.shape[0]
        pi_sqrt = np.sqrt(stationary_distribution)
        pi_inv_sqrt = 1.0 / pi_sqrt
        pi_sqrt = sp.diags(pi_sqrt)
        pi_inv_sqrt = sp.diags(pi_inv_sqrt)

        s = pi_sqrt @ transition_matrix @ pi_inv_sqrt
        s_t = pi_inv_sqrt @ transition_matrix.T @ pi_sqrt
        random_walk_laplacian = sp.eye(n) - 0.5 * (s + s_t)

        return random_walk_laplacian

    def _spectral_clustering(self, adjacency_matrix: sp.csr_matrix):
        n = adjacency_matrix.shape[0]
        M = self._transition_matrix(adjacency_matrix)
        pi = self._power_iter(M, self._alpha)
        L_rw = self._rw_laplacian(M, pi)

        sym_error = np.max(np.abs(L_rw - L_rw.T))
        assert np.isclose(sym_error, 0, rtol=self._epsilon)

        max_k = self._k if self._k is not None and self._k > 1 else n - 1
        max_eigenvector_count = min(max_k, n - 1)
        eigenvalues, eigenvectors = splinalg.eigsh(
            L_rw, k=max_eigenvector_count, which="SM"
        )

        if max_k >= 2:
            gaps = np.diff(eigenvalues[:max_k])
            k = np.argmax(gaps[1:]) + 2
        else:
            gaps = []
            k = 1

        embedding = eigenvectors[:, :k]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embedding /= norms

        labels = KMeans(n_clusters=k).fit_predict(embedding)

        return labels, eigenvalues, eigenvectors, embedding, pi, gaps

    def _extract_graph_clusters(self, g: nx.Graph) -> Iterable[Iterable[T]]:
        nodes = list(g.nodes)
        adj_matrix = nx.to_scipy_sparse_array(g)
        lbl, lambda_, v, emb, pi, gaps = self._spectral_clustering(adj_matrix)
        clusters = {}
        for idx, cluster_label in enumerate(lbl):
            clusters.setdefault(cluster_label, []).append(nodes[idx])
        for cluster_no, cluster in clusters.items():
            yield cluster

    def __call__(self, similarity_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = self._to_directed(similarity_graph, self._threshold)
        components = list(nx.weakly_connected_components(g))
        clusters = []
        for c in components:
            clusters.extend(self._extract_graph_clusters(g.subgraph(c)))

        return self._add_singletons(self._items, clusters)
