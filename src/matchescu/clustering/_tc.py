from typing import TypeVar, Hashable, Generic
import networkx as nx

from matchescu.reference_store.comparison_space import BinaryComparisonSpace


T = TypeVar("T", bound=Hashable)


class TransitiveClosure(Generic[T]):
    def __init__(self, comparison_space: BinaryComparisonSpace):
        self._items = set(ref_id for pair in comparison_space for ref_id in pair)

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        g = nx.DiGraph()
        g.add_nodes_from(self._items)
        g.add_edges_from(matches)

        tc = nx.transitive_closure(g)

        return frozenset(
            frozenset(node for node in component)
            for component in nx.strongly_connected_components(tc)
        )
