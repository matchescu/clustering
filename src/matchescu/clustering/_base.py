import abc
from collections.abc import Iterable
from typing import TypeVar, Hashable, Generic

from matchescu.similarity import ReferenceGraph

T = TypeVar("T", bound=Hashable)


class ClusteringAlgorithm(Generic[T], metaclass=abc.ABCMeta):
    def __init__(self, all_refs: Iterable[T], threshold: float) -> None:
        self._items = list(set(all_refs))
        self._threshold = threshold

    @abc.abstractmethod
    def __call__(self, similarity_graph: ReferenceGraph) -> frozenset[frozenset[T]]:
        pass
