import math
from enum import StrEnum

import networkx as nx
import numpy as np
import scipy.sparse as sp
from typing import Iterable, Generator
from matchescu.clustering._base import T, ClusteringAlgorithm
from matchescu.similarity import ReferenceGraph


class SeedStrategy(StrEnum):
    DEGREE = "degree"
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"


class PartitionStrategy(StrEnum):
    PAGERANK = "pagerank"
    BRIDGE = "bridge"


# https://arxiv.org/pdf/2412.03008
class ACLClustering(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        alpha: float = 0.15,
        seed_strategy: SeedStrategy = SeedStrategy.BETWEENNESS,
        partition_strategy: PartitionStrategy = PartitionStrategy.BRIDGE,
        detect_scc: bool = True,
        betweenness_sample_count: int | None = None,  # autoselect
    ):
        super().__init__(all_refs, threshold)
        self._alpha = alpha
        self._detect_scc = detect_scc
        self._strategy = (
            self._global_acl_pagerank
            if partition_strategy == PartitionStrategy.PAGERANK
            else self._global_acl_bridge
        )
        self._seed_strategy = seed_strategy
        self._k = betweenness_sample_count

    @staticmethod
    def __build_transition_matrix1(
        digraph: nx.DiGraph, node_indexes: dict[T, int]
    ) -> np.ndarray:
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
    def __build_transition_matrix(
        digraph: nx.DiGraph, node_indexes: dict[T, int]
    ) -> sp.csr_matrix:
        # builds a sparse transition matrix P = D^-1 * A.
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
        # patch up sink node positions (row_sum == 0  -->  set diagonal to 1.0)
        sink_indices = np.where(row_sums == 0)[0]
        if len(sink_indices) > 0:
            sink_node_diagonal_values = sp.diags(np.where(row_sums == 0, 1.0, 0.0))
            transition_matrix = transition_matrix + sink_node_diagonal_values
        return transition_matrix

    @staticmethod
    def __stationary_distribution1(
        transition_matrix: np.ndarray, tol: float = 1e-12, max_iter: int = 20000
    ) -> np.ndarray:
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
    def __lazy_ppr1(
        transition_matrix: np.ndarray,
        s: np.ndarray,
        alpha: float = 0.15,
        tol: float = 1e-12,
        max_iter: int = 20000,
    ) -> np.ndarray:
        p = s.copy().astype(float)
        for _ in range(max_iter):
            # p_next = alpha*s + (1-alpha) * p * M, where M = 0.5*(I + P)
            p_next = alpha * s + (1.0 - alpha) * 0.5 * (p + p.dot(transition_matrix))
            if np.linalg.norm(p_next - p, 1) < tol:
                return p_next
            p = p_next
        return p

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
            # efficient sparse matrix multiplication
            p_next_step = p @ transition_matrix

            # ensure 1D shape
            if p_next_step.ndim > 1:
                p_next_step = np.asarray(p_next_step).flatten()

            # p_next = alpha * s + (1-alpha) * 0.5 * (p + pP)
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
        # handle degenerate volume cases
        if vol_s == 0.0:
            return 1.0

        # vectorize: compute the probability of moving INTO set S from any node
        # multiply transition matrix and indicator vector
        indicator_vector = mask.astype(float)
        prob_to_in = transition_matrix @ indicator_vector

        # Ensure 1D shape
        if prob_to_in.ndim > 1:
            prob_to_in = np.asarray(prob_to_in).flatten()

        # Cut = sum_{u in S} phi[u] * (1 - P(u -> S))
        #     = sum_{u in S} phi[u] * P(u -> not S)
        cut = (phi[mask] * (1.0 - prob_to_in[mask])).sum()
        denominator = min(vol_s, 1.0 - vol_s)
        # defensive if lines above change
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

        # Ensure phi is 1D before boolean indexing
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
        # Ensure page_ranks is 1D
        if page_ranks.ndim > 1:
            page_ranks = np.asarray(page_ranks).flatten()

        denom = phi.copy()
        denom[denom == 0.0] = 1e-30
        score = page_ranks / denom
        order = np.argsort(-score)

        best_cond = float("inf")
        best_set = None
        cur_mask = np.zeros(node_count, dtype=bool)

        # Sweep optimization: The bottleneck is often here.
        # Calling _measure_conductance inside the loop is correct but must be fast.
        for j in range(node_count):
            cur_mask[order[j]] = True
            # Optional: Add early stop if conductance is "good enough" (e.g., < 0.01)
            # to further match the paper's suggestion.
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

    def _compute_centrality(self, g: nx.DiGraph):
        match self._seed_strategy:
            case SeedStrategy.PAGERANK:
                return nx.pagerank(g, weight="weight")
            case SeedStrategy.DEGREE:
                return nx.degree_centrality(g)
            case SeedStrategy.BETWEENNESS:
                k = int(self._k or math.sqrt(len(g.nodes)))
                return nx.betweenness_centrality(g, k, normalized=True, weight="weight")

    def _partition_component(
        self, subgraph: nx.DiGraph, alpha: float
    ) -> Generator[tuple[list[T], float], None, None]:
        """Partition a single structural component using ACL with pagerank seeding."""
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
        """Iteratively selects highest-pagerank unassigned node as seed."""
        yield from self._partition_component(digraph, alpha)

    def _global_acl_bridge(
        self, digraph: nx.DiGraph, alpha: float
    ) -> Generator[tuple[list[T], float], None, None]:
        """Use the undirected skeleton to identify structurally cohesive groups.

        This captures:
        - Rings: a→b→c→d→a (strongly connected cycles)
        - Diamonds: a→b→d, a→c→d (multi-path structures)
        - Cliques: fully connected subgraphs

        Nodes within the same bridge component have 2+ edge-disjoint paths
        between them in the undirected view, indicating structural cohesion.
        """
        undirected = digraph.to_undirected()

        # Find 2-edge-connected components (bridge components)
        # These are maximal subgraphs where no single edge removal disconnects them
        bridge_components = list(nx.connectivity.bridge_components(undirected))

        for component in bridge_components:
            if len(component) <= 1:
                yield list(component), 0.0
                continue

            subgraph = digraph.subgraph(component).to_directed().copy()

            # If SCC shortcut enabled and strongly connected, keep as one cluster
            if self._detect_scc and nx.is_strongly_connected(subgraph):
                yield list(component), 0.0
                continue

            # Check if component is "bridge-free" in undirected sense
            # (i.e., every pair is connected via 2+ paths)
            # If so, keep together as the multi-path structure is cohesive
            component_undirected = undirected.subgraph(component)
            if nx.is_connected(component_undirected) and not list(
                nx.bridges(component_undirected)
            ):
                yield list(component), 0.0
                continue

            # Otherwise, run ACL-based partitioning within this component
            yield from self._partition_component(subgraph, alpha)

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        g = nx.DiGraph()
        for node_u, node_v in reference_graph.matches(self._threshold):
            g.add_edge(node_u, node_v, weight=reference_graph.weight(node_u, node_v))
        singletons = frozenset(self._items) - frozenset(g.nodes)
        clusters = set(frozenset(c) for c, _ in self._strategy(g, self._alpha))
        for singleton in singletons:
            clusters.add(frozenset([singleton]))
        return frozenset(clusters)
