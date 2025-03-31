from typing import Generic, TypeVar
import random

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph
from matchescu.typing import EntityReferenceIdentifier

T = TypeVar("T", bound=EntityReferenceIdentifier)


class WeightedCorrelationClustering(Generic[T]):
    def __init__(
        self,
        all_comparisons: BinaryComparisonSpace,
        threshold: float = 0.0,
        random_seed: int | None = None,
    ) -> None:
        self._items = set(item for pair in all_comparisons for item in pair)
        if random_seed:
            random.seed(random_seed)
        self._threshold = threshold

    def __call__(self, similarity_graph: SimilarityGraph) -> frozenset[frozenset[T]]:
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
                    if similarity_graph.weight(pivot, node) >= self._threshold
                )
            )

            all_clusters.append(pivot_cluster)
            unclustered_nodes -= pivot_cluster

        return frozenset(all_clusters)
