import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
from enum import StrEnum
from typing import Iterable, Generator
from matchescu.clustering._base import (
    T,
    ClusteringAlgorithm,
    SingletonHandlerMixin,
    NxDirectedMixin,
)
from matchescu.similarity import ReferenceGraph


class SeedStrategy(StrEnum):
    """Strategy for selecting seed nodes in ACL."""

    BETWEENNESS = "betweenness"
    BETWEENNESS_SAMPLED = "betweenness_sampled"
    PAGERANK = "pagerank"
    DEGREE = "degree"


class PartitionStrategy(StrEnum):
    """Strategy for global graph partitioning."""

    PAGERANK = "pagerank"
    DIRECTED = "directed"


# https://arxiv.org/pdf/2412.03008
class ACLClustering(ClusteringAlgorithm[T], SingletonHandlerMixin[T], NxDirectedMixin):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        alpha: float = 0.15,
        seed_strategy: SeedStrategy = SeedStrategy.BETWEENNESS_SAMPLED,
        partition_strategy: PartitionStrategy = PartitionStrategy.DIRECTED,
        detect_scc: bool = True,
        betweenness_sample_count: int | None = None,
    ):
        super().__init__(all_refs, threshold)
        self._alpha = alpha
        self._detect_scc = detect_scc
        match partition_strategy:
            case PartitionStrategy.PAGERANK:
                self._strategy = self._global_acl_pagerank
            case _:
                self._strategy = self._global_acl_directed
        self._seed_strategy = seed_strategy
        self._k = betweenness_sample_count

    @staticmethod
    def __build_transition_matrix(
        digraph: nx.DiGraph, node_indexes: dict[T, int]
    ) -> sp.csr_matrix:
        nodelist = list(node_indexes.keys())
        adjacency = nx.to_scipy_sparse_array(
            digraph, nodelist=nodelist, weight="weight", format="csr"
        )
        row_sums = np.array(adjacency.sum(axis=1)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_vals = 1.0 / row_sums
        d_inv_vals[np.isinf(d_inv_vals)] = 0.0
        inverse_diag = sp.diags(d_inv_vals)

        transition_matrix = inverse_diag @ adjacency
        sink_indices = np.where(row_sums == 0)[0]
        if len(sink_indices) > 0:
            sink_node_diagonal_values = sp.diags(np.where(row_sums == 0, 1.0, 0.0))
            transition_matrix = transition_matrix + sink_node_diagonal_values
        return transition_matrix

    @staticmethod
    def __stationary_distribution(
        transition_matrix: sp.csr_matrix, tol: float = 1e-12, max_iter: int = 20000
    ) -> np.ndarray:
        n = transition_matrix.shape[0]
        phi = np.ones(n) / n
        for _ in range(max_iter):
            phi_next = phi @ transition_matrix
            if phi_next.ndim > 1:
                phi_next = np.asarray(phi_next).flatten()
            s = phi_next.sum()
            phi_next = np.ones(n) / n if s == 0 else phi_next / s
            if np.linalg.norm(phi_next - phi, 1) < tol:
                return phi_next
            phi = phi_next
        return phi

    @staticmethod
    def __lazy_ppr(
        transition_matrix: sp.csr_matrix,
        s: np.ndarray,
        alpha: float = 0.15,
        tol: float = 1e-12,
        max_iter: int = 20000,
    ) -> np.ndarray:
        p = s.copy().astype(float)
        factor = (1.0 - alpha) * 0.5
        for _ in range(max_iter):
            p_next_step = p @ transition_matrix
            if p_next_step.ndim > 1:
                p_next_step = np.asarray(p_next_step).flatten()
            p_next = alpha * s + factor * (p + p_next_step)
            if np.linalg.norm(p_next - p, 1) < tol:
                return p_next
            p = p_next
        return p

    @staticmethod
    def _measure_conductance(
        mask: np.ndarray, transition_matrix: sp.csr_matrix, phi: np.ndarray
    ) -> float:
        n = transition_matrix.shape[0]
        count = mask.sum()
        if count == 0 or count == n:
            return 1.0
        vol_s = phi[mask].sum()
        if vol_s == 0.0:
            return 1.0
        indicator_vector = mask.astype(float)
        prob_to_in = transition_matrix @ indicator_vector
        if prob_to_in.ndim > 1:
            prob_to_in = np.asarray(prob_to_in).flatten()
        cut = (phi[mask] * (1.0 - prob_to_in[mask])).sum()
        denominator = min(vol_s, 1.0 - vol_s)
        return cut / denominator if denominator > 0 else 1.0

    def _general_acl(
        self,
        digraph: nx.DiGraph,
        seeds: Iterable[T],
        alpha: float = 0.15,
        tol: float = 1e-12,
        max_iter: int = 20000,
    ) -> tuple[list[str], float]:
        nodes = list(digraph.nodes())
        if self._detect_scc and nx.is_strongly_connected(digraph):
            return nodes, 0.0

        node_index = {node_val: i for i, node_val in enumerate(nodes)}
        node_count = len(node_index)

        transition_matrix = self.__build_transition_matrix(digraph, node_index)
        sparse_identity = sp.eye(node_count, format="csr")
        lazy_walk_input = 0.5 * (sparse_identity + transition_matrix)

        phi = self.__stationary_distribution(
            lazy_walk_input, tol=tol, max_iter=max_iter
        )
        if phi.ndim > 1:
            phi = np.asarray(phi).flatten()

        mask = np.zeros(node_count, dtype=bool)
        for seed in seeds:
            mask[node_index[seed]] = True

        volS = phi[mask].sum()
        if volS == 0.0:
            seeds = self._handle_zero_volume(digraph, seeds)
            mask = np.zeros(node_count, dtype=bool)
            for seed in seeds:
                mask[node_index[seed]] = True
            volS = phi[mask].sum()

        psi = np.zeros(node_count, dtype=float)
        psi[mask] = phi[mask] / volS

        page_ranks = self.__lazy_ppr(
            transition_matrix, psi, alpha=alpha, tol=tol, max_iter=max_iter
        )
        if page_ranks.ndim > 1:
            page_ranks = np.asarray(page_ranks).flatten()

        denom = phi.copy()
        denom[denom == 0.0] = 1e-30
        score = page_ranks / denom
        order = np.argsort(-score)

        best_cond = float("inf")
        best_set = None
        cur_mask = np.zeros(node_count, dtype=bool)

        for j in range(node_count):
            cur_mask[order[j]] = True
            cond = self._measure_conductance(cur_mask, transition_matrix, phi)
            if cond < best_cond:
                best_cond = cond
                best_set = cur_mask.copy()

        best_nodes = [nodes[i] for i in np.where(best_set)[0]]
        return best_nodes, best_cond

    @staticmethod
    def _handle_zero_volume(digraph: nx.DiGraph, seeds: Iterable[T]):
        reachable = set()
        for seed in seeds:
            reachable.update(nx.descendants(digraph, seed))
        return list(set(seeds) | reachable)

    def _compute_centrality(self, g: nx.DiGraph) -> dict[T, float]:
        """Compute node centrality based on selected strategy."""
        match self._seed_strategy:
            case SeedStrategy.BETWEENNESS:
                return nx.betweenness_centrality(g, weight="weight")
            case SeedStrategy.BETWEENNESS_SAMPLED:
                n = g.number_of_nodes()
                k = min(n, int(self._k or max(3, min(10, int(math.sqrt(n))))))
                return nx.betweenness_centrality(g, k=k, weight="weight")
            case SeedStrategy.PAGERANK:
                return nx.pagerank(g, weight="weight")
            case SeedStrategy.DEGREE:
                return nx.degree_centrality(g)

    def _partition_component(
        self, subgraph: nx.DiGraph, alpha: float
    ) -> Generator[tuple[list[T], float], None, None]:
        """Partition a single structural component using ACL."""
        centrality = self._compute_centrality(subgraph)
        sorted_nodes = sorted(centrality, key=lambda x: -centrality[x])

        assigned = set()
        for node in sorted_nodes:
            if node in assigned:
                continue
            sub_nodes = [n for n in subgraph.nodes() if n not in assigned]
            if not sub_nodes:
                break
            inner_subgraph = subgraph.subgraph(sub_nodes).to_directed().copy()
            if node not in inner_subgraph:
                continue

            cluster, cond = self._general_acl(inner_subgraph, [node], alpha=alpha)
            cluster_set = set(cluster)
            assigned.update(cluster_set)
            yield cluster, cond

    def _global_acl_pagerank(
        self, digraph: nx.DiGraph, alpha: float
    ) -> Generator[tuple[list[T], float], None, None]:
        """Original approach: centrality-based seed selection."""
        yield from self._partition_component(digraph, alpha)

    def _global_acl_directed(
        self,
        digraph: nx.DiGraph,
        alpha: float,
    ) -> Generator[tuple[list[T], float], None, None]:
        """
        Directed-aware clustering via reciprocal contraction + ACL.

        1. Find connected components of reciprocal edges (A↔B).
           These are hard "must-link" constraints.
        2. Contract each reciprocal group into a single super-node.
           Cross-group edges keep the max weight.
        3. Run iterative ACL (_partition_component) on the contracted graph.
        4. Expand super-nodes back to original nodes.

        This guarantees reciprocal pairs always cluster together,
        while letting ACL decide on one-way and structural connections.
        When no reciprocal edges exist, this is identical to PAGERANK strategy.
        """
        # Step 1: Build reciprocal graph (undirected edges where both directions exist)
        recip = nx.Graph()
        recip.add_nodes_from(digraph.nodes())
        for u, v, w in digraph.edges.data("weight"):
            w_rev = digraph.get_edge_data(v, u, {}).get("weight", 0)
            if w > self._threshold and w_rev > self._threshold:
                recip.add_edge(u, v)

        # Connected components of reciprocal graph = "must-link" groups
        recip_groups = list(nx.connected_components(recip))

        # Step 2: Map each node to a group representative
        node_to_rep: dict[T, T] = {}
        rep_to_nodes: dict[T, list[T]] = {}
        for group in recip_groups:
            group_list = sorted(group, key=repr)
            rep = group_list[0]
            rep_to_nodes[rep] = group_list
            for node in group_list:
                node_to_rep[node] = rep

        # Step 3: Build contracted directed graph
        contracted = nx.DiGraph()
        for rep in rep_to_nodes:
            contracted.add_node(rep)

        for u, v, data in digraph.edges(data=True):
            rep_u, rep_v = node_to_rep[u], node_to_rep[v]
            if rep_u == rep_v:
                continue  # intra-group edge, already clustered
            w = data.get("weight", 1.0)
            if contracted.has_edge(rep_u, rep_v):
                # Keep the strongest cross-group signal
                contracted[rep_u][rep_v]["weight"] = max(
                    contracted[rep_u][rep_v]["weight"], w
                )
            else:
                contracted.add_edge(rep_u, rep_v, weight=w)

        # Step 4: Partition contracted graph with iterative ACL
        for cluster_reps, cond in self._partition_component(contracted, alpha):
            # Step 5: Expand representatives back to original nodes
            expanded = []
            for rep in cluster_reps:
                expanded.extend(rep_to_nodes[rep])
            yield expanded, cond

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = self._to_directed(reference_graph, self._threshold)
        clusters = set(frozenset(c) for c, _ in self._strategy(g, self._alpha))
        return self._add_singletons(self._items, clusters)
