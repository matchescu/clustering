from functools import reduce
from typing import Generator, Iterable, Callable, Optional

from pandas import DataFrame
from ppjoin import ppjoin

from matchescu.adt.entity_resolution_result import EntityResolutionResult
from matchescu.adt.types import Record
from matchescu.common.partitioning import compute_partition


def _compute_fsm(input_data: list[DataFrame], result: Iterable[tuple[tuple]]) -> Generator[tuple, None, None]:
    for r in result:
        ds1_id, r1id = r[0]
        ds2_id, r2id = r[1]
        item1 = tuple(v for v in input_data[ds1_id].iloc[r1id, :])
        item2 = tuple(v for v in input_data[ds2_id].iloc[r2id, :])
        yield item1, item2


def _compute_partition(input_data: list[DataFrame], result: Iterable[tuple[tuple]]) -> frozenset:
    partition = dict()
    for r in result:
        ds1_id, r1id = r[0]
        ds2_id, r2id = r[1]
        item1 = tuple(v for v in input_data[ds1_id].iloc[r1id, :])
        item2 = tuple(v for v in input_data[ds2_id].iloc[r2id, :])
        cluster = (item1, item2)
        partition[cluster] = partition.get(cluster, 0) + 1
    return frozenset(partition)


def _compute_algebraic(partition: frozenset[tuple[tuple]]) -> Generator[tuple, None, None]:
    for item in partition:
        yield item


def _compute_serf(
    partition: frozenset[tuple[tuple]],
    merge_function: Callable[[Record, Record], Record],
) -> Generator[tuple, None, None]:
    for cluster in partition:
        merged_eq_class = reduce(merge_function, cluster)
        yield merged_eq_class


def ppjoin_adapter(
    input_data: list[DataFrame],
    threshold: float,
    merge_function: Optional[Callable[[Record, Record], Record]] = None,
) -> EntityResolutionResult:
    ds = [
        [
            [v for v in set(str(value).lower() for value in row)]
            for index, row in dataframe.iterrows()
        ]
        for dataframe in input_data
    ]
    result = ppjoin.join(ds, t=threshold)

    er_result = EntityResolutionResult()
    er_result.fsm = list(_compute_fsm(input_data, result))
    partition = _compute_partition(input_data, result)
    er_result.algebraic = list(_compute_algebraic(partition))
    if merge_function is not None:
        er_result.serf = list(_compute_serf(partition, merge_function))

    return er_result
