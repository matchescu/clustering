import networkx as nx
import numpy as np
from typing import Iterable, Optional, Generator
from matchescu.clustering._base import T, ClusteringAlgorithm
from matchescu.similarity import ReferenceGraph


class ACLClustering(ClusteringAlgorithm[T]):
    def __init__(self, all_refs: Iterable[T], threshold: float = 0.75, alpha: float = 0.15):
        super().__init__(all_refs, threshold)
        self._alpha = alpha

    @staticmethod
    def __build_transition_matrix(digraph: nx.DiGraph, node_indexes: dict[T, int]) -> np.ndarray:
        n = len(node_indexes)
        result = np.zeros((n, n), dtype=float)
        for node, i in node_indexes.items():
            out_sum = 0.0
            for next_node in digraph.successors(node):
                w = digraph[node][next_node].get("weight", 1.0)
                out_sum += w
                result[i, node_indexes[next_node]] = w
            if out_sum == 0.0:
                result[i, i] = 1.0
            else:
                result[i, :] /= out_sum
        return result

    @staticmethod
    def __stationary_distribution(transition_matrix: np.ndarray, tol: float = 1e-12, max_iter: int = 20000) -> np.ndarray:
        n = transition_matrix.shape[0]
        phi = np.ones(n) / n
        for _ in range(max_iter):
            phi_next = phi.dot(transition_matrix)
            s = phi_next.sum()
            if s == 0:
                phi_next = np.ones(n) / n
            else:
                phi_next /= s
            if np.linalg.norm(phi_next - phi, 1) < tol:
                return phi_next
            phi = phi_next
        return phi

    @staticmethod
    def __lazy_ppr(transition_matrix: np.ndarray, s: np.ndarray, alpha: float = 0.15,
                   tol: float = 1e-12, max_iter: int = 20000) -> np.ndarray:
        p = s.copy().astype(float)
        for _ in range(max_iter):
            # p_next = alpha*s + (1-alpha) * p * M, where M = 0.5*(I + P)
            p_next = alpha * s + (1.0 - alpha) * 0.5 * (p + p.dot(transition_matrix))
            if np.linalg.norm(p_next - p, 1) < tol:
                return p_next
            p = p_next
        return p

    @staticmethod
    def _measure_conductance(mask: np.ndarray, transition_matrix: np.ndarray, phi: np.ndarray) -> float:
        if mask.sum() == 0 or mask.sum() == transition_matrix.shape[0]:
            return 1.0
        P_in_S = transition_matrix[:, mask]
        prob_to_in = P_in_S.sum(axis=1)
        cut = (phi[mask] * (1.0 - prob_to_in[mask])).sum()
        volS = phi[mask].sum()
        denom = min(volS, 1.0 - volS)
        return cut / denom if denom > 0 else 1.0

    @classmethod
    def _general_acl(cls, digraph: nx.DiGraph, seeds: Iterable[T], alpha: float = 0.15,
                     tol: float = 1e-12, max_iter: int = 20000) -> tuple[list[str], float]:
        nodes = list(digraph.nodes())
        node_index = {node_val: i for i, node_val in enumerate(nodes)}
        node_count = len(node_index)
        transition_matrix = cls.__build_transition_matrix(digraph, node_index)

        lazy_walk_input = 0.5 * (np.eye(node_count) + transition_matrix)
        phi = cls.__stationary_distribution(lazy_walk_input, tol=tol, max_iter=max_iter)
        mask = np.zeros(node_count, dtype=bool)
        for seed in seeds:
            mask[node_index[seed]] = True
        volS = phi[mask].sum()
        # If volS is zero, adjust seeds using _handle_zero_volume
        if volS == 0.0:
            seeds = cls._handle_zero_volume(digraph, seeds)
            mask = np.zeros(node_count, dtype=bool)
            for seed in seeds:
                mask[node_index[seed]] = True
            volS = phi[mask].sum()

        psi = np.zeros(node_count, dtype=float)
        psi[mask] = phi[mask] / volS
        page_ranks = cls.__lazy_ppr(transition_matrix, psi, alpha=alpha, tol=tol, max_iter=max_iter)

        denom = phi.copy()
        denom[denom == 0.0] = 1e-30
        score = page_ranks / denom
        order = np.argsort(-score)

        best_cond = float('inf')
        best_set = None
        cur_mask = np.zeros(node_count, dtype=bool)
        for j in range(node_count):
            cur_mask[order[j]] = True
            cond = cls._measure_conductance(cur_mask, transition_matrix, phi)
            if cond < best_cond:
                best_cond = cond
                best_set = cur_mask.copy()
        best_nodes = [nodes[i] for i in np.where(best_set)[0]]
        return best_nodes, best_cond

    @staticmethod
    def _handle_zero_volume(digraph: nx.DiGraph, seeds: Iterable[T]):
        # Collect all descendants from each seed
        reachable = set()
        for seed in seeds:
            reachable.update(nx.descendants(digraph, seed))
        return list(set(seeds) | reachable)

    @classmethod
    def _global_acl(cls, digraph: nx.DiGraph, alpha: float = 0.15) -> Generator[tuple[list[T], float], None, None]:
        """
        Partition the graph into clusters by repeatedly running general_acl from seeds chosen by betweenness centrality.
        Each node is assigned to at most one cluster.
        Returns a list of (cluster_nodes, conductance) tuples.
        """
        # Compute betweenness centrality (on directed, weighted graph)
        centrality = nx.betweenness_centrality(digraph, weight='weight', normalized=True)
        # Sort nodes descending by centrality
        sorted_nodes = sorted(centrality, key=lambda x: -centrality[x])
        assigned = set()
        for node in sorted_nodes:
            if node in assigned:
                continue
            sub_nodes = [n for n in digraph.nodes() if n not in assigned]
            subgraph = digraph.subgraph(sub_nodes).copy().to_directed()
            cluster, cond = cls._general_acl(subgraph, [node], alpha=alpha)
            cluster_set = set(cluster)
            assigned.update(cluster_set)
            yield cluster, cond

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = nx.DiGraph()

        for node_u, node_v in reference_graph.matches(self._threshold):
            g.add_edge(node_u, node_v, weight=reference_graph.weight(node_u, node_v))

        result = frozenset(frozenset(c) for c, _ in self._global_acl(g, self._alpha))
        return result
