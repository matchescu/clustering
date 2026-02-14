import abc
from collections.abc import Iterable
from typing import TypeVar, Hashable, Generic

import networkx as nx

from matchescu.similarity import ReferenceGraph

T = TypeVar("T", bound=Hashable)


class ClusteringAlgorithm(Generic[T], metaclass=abc.ABCMeta):
    def __init__(self, all_refs: Iterable[T], threshold: float) -> None:
        self._items = list(set(all_refs))
        self._threshold = threshold

    @abc.abstractmethod
    def __call__(self, similarity_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        pass


class NxDirectedMixin:
    @classmethod
    def _to_directed(cls, reference_graph: ReferenceGraph, t: float) -> nx.DiGraph:
        g = nx.DiGraph()
        for u, v in reference_graph.matches(t):
            g.add_edge(u, v, weight=reference_graph.weight(u, v))
        return g


class SingletonHandlerMixin(Generic[T]):
    @classmethod
    def _add_singletons(
        cls, items: Iterable[T], clusters: Iterable[Iterable[T]]
    ) -> frozenset[frozenset[T]]:
        unassigned = items
        partition = []
        if clusters is not None:
            assigned = set(node for cluster in clusters for node in cluster)
            unassigned = set(items) - assigned
            partition = [frozenset(node for node in cluster) for cluster in clusters]
        partition.extend(frozenset([ref]) for ref in unassigned)
        return frozenset(cluster for cluster in partition)
