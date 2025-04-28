import random
from collections.abc import Iterable

from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import T, ClusteringAlgorithm


class WeightedCorrelationClustering(ClusteringAlgorithm[T]):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(all_refs, threshold)
        if random_seed:
            random.seed(random_seed)

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        unclustered_nodes = set(self._items)
        all_clusters = []

        while unclustered_nodes:
            pivot = random.choice(list(unclustered_nodes))
            nodes_to_check = list(unclustered_nodes - {pivot})
            random.shuffle(nodes_to_check)

            pivot_cluster = frozenset(
                {pivot}
                | set(
                    node
                    for node in nodes_to_check
                    if reference_graph.weight(pivot, node) >= self._threshold
                )
            )

            all_clusters.append(pivot_cluster)
            unclustered_nodes -= pivot_cluster

        return frozenset(all_clusters)
