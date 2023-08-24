import itertools
from typing import Generator, Iterable, Callable, Optional

from pandas import DataFrame
from ppjoin import ppjoin

from matchescu.entity_resolution_result import EntityResolutionResult
from matchescu.types import Record


def _compute_fsm(input_data: list[DataFrame], result: Iterable[tuple[tuple]]) -> Generator[tuple, None, None]:
    for r in result:
        ds1_id, r1id = r[0]
        ds2_id, r2id = r[1]
        item1 = tuple(v for v in input_data[ds1_id].iloc[r1id, :])
        item2 = tuple(v for v in input_data[ds2_id].iloc[r2id, :])
        yield item1, item2


def _compute_algebraic(input_data: list[DataFrame], result: Iterable[tuple[tuple]]) -> Generator[tuple, None, None]:
    partition = {}
    for r in result:
        ds1_id, i = r[0]
        ds2_id, j = r[1]
        item1 = tuple(v for v in input_data[ds1_id].iloc[i, :])
        item2 = tuple(v for v in input_data[ds2_id].iloc[j, :])
        partition[r[0]] = partition.get(r[0], set()) | {item1, item2}
        partition[r[1]] = partition.get(r[1], set()) | {item1, item2}

    # eliminate duplicate partition classes
    partition = {
        tuple(v for v in value): key
        for key, value in partition.items()
    }

    for partition_class in partition:
        yield partition_class


def _compute_serf(
    input_data: list[DataFrame],
    result: Iterable[tuple[tuple]],
    merge_function: Callable[[Record, Record], Record],
):
    serf_results = []
    found = set()
    for r in result:
        ds1_id, r1id = r[0]
        ds2_id, r2id = r[1]
        df1 = input_data[ds1_id]
        df2 = input_data[ds2_id]
        row0 = tuple(v for v in (df1.iloc[r1id, :]))
        row1 = tuple(v for v in (df2.iloc[r2id, :]))
        found |= {r[0], r[1]}
        serf_results.append(
            tuple(value for value in merge_function(row0, row1))
        )
    for df_idx, dataframe in enumerate(input_data):
        for row_idx in dataframe.index:
            if (df_idx, row_idx) not in found:
                record = tuple(v for v in dataframe.loc[row_idx, :])
                serf_results.append((record,))
    for result in serf_results:
        yield result


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
    er_result.algebraic = list(_compute_algebraic(input_data, result))
    if merge_function is not None:
        er_result.serf = list(_compute_serf(input_data, result, merge_function))

    return er_result
