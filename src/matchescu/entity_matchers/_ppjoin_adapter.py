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


def _get_partitioned_set(input_data: Iterable[DataFrame]) -> Generator[tuple, None, None]:
    yield from (
        (df_idx, idx)
        for df_idx, df in enumerate(input_data)
        for idx in df.index
    )


def _map_to_data(input_data: list[DataFrame]) -> Callable[[Iterable[tuple]], Iterable[tuple]]:
    def _map(equivalence_class: Iterable[tuple]) -> Iterable[tuple]:
        ret = []
        for df_id, row_id in equivalence_class:
            row = tuple(value for value in input_data[df_id].iloc[row_id, :])
            ret.append(row)
        return ret
    return _map


def _compute_algebraic(input_data: list[DataFrame], partition: set) -> Generator[tuple, None, None]:
    yield from map(_map_to_data(input_data), partition)


def _compute_serf(
    input_data: list[DataFrame],
    partition: set,
    merge_function: Callable[[Record, Record], Record],
):
    for eq_class in map(_map_to_data(input_data), partition):
        merged_eq_class = reduce(merge_function, eq_class)
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
    # partition = compute_partition(list(_get_partitioned_set(input_data)), result)
    # er_result.algebraic = list(_compute_algebraic(input_data, partition))
    # if merge_function is not None:
    #     er_result.serf = list(_compute_serf(input_data, partition, merge_function))

    return er_result
