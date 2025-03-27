from typing import TypeVar, Hashable, Generic

from matchescu.reference_store.comparison_space import BinaryComparisonSpace


T = TypeVar("T", bound=Hashable)


class EquivalenceClassPartitioner(Generic[T]):
    def __init__(self, comparison_space: BinaryComparisonSpace):
        self._items = list(set(ref_id for pair in comparison_space for ref_id in pair))
        self._rank = {item: 0 for item in self._items}
        self._parent = {item: item for item in self._items}

    def _find(self, x: T) -> T:
        if self._parent[x] == x:
            return x
        # path compression
        self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

    def _union(self, x: T, y: T) -> None:
        x_root = self._find(x)
        y_root = self._find(y)

        if x_root == y_root:
            return

        if self._rank[x_root] < self._rank[y_root]:
            self._parent[x_root] = y_root
        elif self._rank[y_root] < self._rank[x_root]:
            self._parent[y_root] = x_root
        else:
            # does not matter which goes where
            # make sure we increase the correct rank
            self._parent[y_root] = x_root
            self._rank[x_root] += 1

    def __call__(self, matches: list[tuple[T, T]]) -> frozenset[frozenset[T]]:
        for x, y in matches:
            self._union(x, y)
        classes = {item: dict() for item in self._items}
        for item in self._items:
            classes[self._find(item)][item] = None
        return frozenset(
            frozenset(eq_class) for eq_class in classes.values()
            if len(eq_class) > 0
        )
