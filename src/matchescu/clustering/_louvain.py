from collections.abc import Iterable

import networkx.algorithms.community.louvain as nx_louvain
from matchescu.similarity import ReferenceGraph

from matchescu.clustering._base import (
    T,
    ClusteringAlgorithm,
    NxDirectedMixin,
    SingletonHandlerMixin,
)


class LouvainPartitioning(
    ClusteringAlgorithm[T], SingletonHandlerMixin[T], NxDirectedMixin
):
    def __init__(
        self,
        all_refs: Iterable[T],
        threshold: float = 0.75,
        louvain_resolution: float = 1,
        louvain_threshold: float = 0.0000001,
    ) -> None:
        super().__init__(all_refs, threshold)
        self._alg_resolution = louvain_resolution
        self._alg_threshold = louvain_threshold

    def __call__(self, reference_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        graph = self._to_directed(reference_graph, self._threshold)

        best = None
        for partition in nx_louvain.louvain_partitions(
            graph,
            weight="weight",
            threshold=self._alg_threshold,
            resolution=self._alg_resolution,
        ):
            best = partition

        return self._add_singletons(self._items, best)
