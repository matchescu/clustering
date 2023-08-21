from typing import Generator, Iterable

from pandas import DataFrame
from ppjoin import ppjoin

from matchescu.entity_resolution_result import EntityResolutionResult


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


def ppjoin_adapter(input_data: list[DataFrame]) -> EntityResolutionResult:
    ds = [
        [
            [v for v in set(str(value).lower() for value in row)]
            for index, row in dataframe.iterrows()
        ]
        for dataframe in input_data
    ]
    result = ppjoin.join(ds, t=0.5)

    er_result = EntityResolutionResult()
    er_result.fsm = list(_compute_fsm(input_data, result))
    er_result.algebraic = list(_compute_algebraic(input_data, result))

    return er_result
