import re
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


def _compute_partition(input_data: list[DataFrame], result: Iterable[tuple[tuple]]) -> list[list[tuple]]:
    pairs = {}
    for r in result:
        ds1_id, r1id = r[0]
        ds2_id, r2id = r[1]
        item1 = tuple(v for v in input_data[ds1_id].iloc[r1id, :])
        item2 = tuple(v for v in input_data[ds2_id].iloc[r2id, :])
        pair = (item1, item2)
        pairs[pair] = None
    ref_domain = {
        tuple(v for v in row): None
        for df in input_data
        for _, row in df.iterrows()
    }
    return compute_partition(list(ref_domain), list(pairs))


def _compute_algebraic(partition: list[list[tuple]]) -> Generator[tuple, None, None]:
    for item in partition:
        yield item


def _compute_serf(
    partition: list[list[tuple]],
    merge_function: Callable[[Record, Record], Record],
) -> Generator[tuple, None, None]:
    for cluster in partition:
        merged_eq_class = reduce(merge_function, cluster)
        yield merged_eq_class


def _extract_words(row: Iterable) -> Generator[str, None, None]:
    for item in row:
        item_string = str(item)
        words = re.split(r"[\b\W]+", item_string)
        for word in words:
            yield word


def _remove_duplicates(dataframe: DataFrame) -> list[list[str]]:
    return list(
        {
            list(_extract_words(row)): index
            for index, row in dataframe.iterrows()
        }
    )


def _extract_entity_references(dataframe: DataFrame) -> list[list[str]]:
    return list(
        list(_extract_words(row))
        for _, row in dataframe.iterrows()
    )


def ppjoin_adapter(
    input_data: list[DataFrame],
    threshold: float,
    merge_function: Optional[Callable[[Record, Record], Record]] = None,
    model: str = "fsm",
) -> EntityResolutionResult:
    extract_references = _extract_entity_references
    if model == "algebraic":
        extract_references = _remove_duplicates

    ds = [
        extract_references(df)
        for df in input_data
    ]
    result = ppjoin.join(ds, t=threshold)

    er_result = EntityResolutionResult()
    er_result.fsm = list(_compute_fsm(input_data, result))
    partition = _compute_partition(input_data, result)
    er_result.algebraic = list(_compute_algebraic(partition))
    if merge_function is not None:
        er_result.serf = list(_compute_serf(partition, merge_function))

    return er_result
