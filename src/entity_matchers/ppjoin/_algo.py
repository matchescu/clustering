from collections import Counter, namedtuple
from math import ceil
from typing import Any, Callable, Iterable, Generator


def prefix_len(x: tuple, t: float) -> int:
    return int(len(x) - ceil(t * len(x)) + 1)


def compute_alpha(x: tuple, y: tuple, t: float) -> float:
    coefficient = t / (1 + t)
    union_norm = len(x) + len(y)
    return ceil(coefficient * union_norm)


def _default_sort_key(token: Any) -> Any:
    return str(token)


def _compute_inverted_index(data: Iterable[tuple], t: float) -> dict[Any, list[int]]:
    ii = {}
    for x, row in enumerate(data):
        for y, value in enumerate(row):
            if y > prefix_len(row, t):
                break
            if value not in ii:
                ii[value] = [x]
            else:
                ii[value].append(x)
    return ii


def _tuple_without_indexes(row: tuple, excluded: set[int]):
    return tuple(item for idx, item in enumerate(row) if idx not in excluded)


def _prepare_data(
    data: list[tuple], sort_key: Callable[[Any], Any], excluded_indexes: set[int]
) -> tuple[dict[int, int], list[tuple]]:
    row_sorting_param = namedtuple("row_sorting_param", ["original_position", "data"])
    sorted_rows = list(
        row_sorting_param(
            original_position=index,
            data=tuple(sorted(_tuple_without_indexes(row, excluded_indexes), key=sort_key)),
        )
        for index, row in enumerate(data)
    )

    row_index_mapping = {}
    multiset = []
    for sorted_index, item in enumerate(sorted(sorted_rows, key=lambda row: len(row.data))):
        multiset.append(item.data)
        row_index_mapping[sorted_index] = item.original_position

    return row_index_mapping, multiset


def _generate_candidates(
    data: list[tuple],
    inverted_index: dict[Any, list[int]],
    t: float
) -> Generator[tuple[int, int], None, None]:
    for i, x in enumerate(data):
        x_prefix_len = prefix_len(x, t)
        for j in range(x_prefix_len):
            for candidate_index in inverted_index[x[j]]:
                if i != candidate_index:
                    yield i, candidate_index


def _add_to_inverted_index(
    index: dict[Any, list[tuple[int, int]]],
    token: Any,
    row_id: int,
    token_position: int
) -> dict[Any, list[tuple[int, int]]]:
    new_element = (row_id, token_position)
    if token not in index:
        index[token] = [new_element]
    else:
        index[token].append(new_element)
    return index


def _verify(
    multiset: list[tuple],
    x_idx: int,
    index_map: dict[int, int],
    t: float,
    sort_key: Callable
) -> Generator[tuple[tuple, tuple, float], None, None]:
    x = multiset[x_idx]
    px = prefix_len(x, t)
    for y_idx in index_map:
        if index_map[y_idx] < 1:
            continue
        y = multiset[y_idx]
        py = prefix_len(y, t)
        alpha = compute_alpha(x, y, t)
        left_overlap = index_map[y_idx]
        overlap = left_overlap
        if sort_key(x[px-1]) < sort_key(y[py-1]):
            ubound = left_overlap + len(x) - px
            if ubound >= alpha:
                x_counter = Counter(x[px:])
                y_counter = Counter(y[left_overlap:])
                overlap += (x_counter & y_counter).total()
        else:
            ubound = left_overlap + len(y) - py
            if ubound >= alpha:
                x_counter = Counter(x[left_overlap:])
                y_counter = Counter(y[py:])
                overlap += (x_counter & y_counter).total()
        if overlap >= alpha:
            yield x_idx, y_idx


def find_duplicates(
    data: list[tuple],
    t: float,
    sort_key: Callable = None,
    exclude_cols: Iterable[int] = None
) -> set[tuple[tuple, tuple, float]]:
    """Compute the similarity between the rows of a multiset."""
    if t < 0 or 1 <= t:
        raise ValueError(
            f"similarity threshold '{t}' must be between [0, 1)."
        )

    sort_key = sort_key or _default_sort_key
    row_index_mapping, multiset = _prepare_data(
        data, sort_key, set(exclude_cols or [])
    )

    successful_candidates = set()
    inverted_index: dict[Any, list[tuple[int, int]]] = {}
    for x_idx, x in enumerate(multiset):
        index_map: dict[int, int] = {}
        for i in range(prefix_len(x, t)):
            token = x[i]
            for y_idx, j in inverted_index.get(token, []):
                if len(multiset[y_idx]) < t*len(x):
                    continue
                y = multiset[y_idx]
                alpha = compute_alpha(x, y, t)
                ubound = 1 + min(len(x)-i, len(y)-j)
                left_overlap = index_map.get(y_idx, 0)
                if left_overlap + ubound >= alpha:
                    index_map[y_idx] = left_overlap + 1
                else:
                    index_map[y_idx] = 0
            _add_to_inverted_index(inverted_index, token, x_idx, i)

        for sx_idx, sy_idx in _verify(multiset, x_idx, index_map, t, sort_key):
            if (sx_idx, sy_idx) in successful_candidates or (sy_idx, sx_idx) in successful_candidates:
                continue
            successful_candidates.add((sx_idx, sy_idx))

    result = set()

    for x_idx, y_idx in successful_candidates:
        result_x = data[row_index_mapping[x_idx]]
        result_y = data[row_index_mapping[y_idx]]
        sx = set(result_x)
        sy = set(result_y)
        similarity = len(sx & sy) / len(sx | sy)
        result.add((result_x, result_y, similarity))

    return result


def _merge_datasets(datasets: Iterable[list[tuple]], id_col_index: int = 0) -> list[tuple]:
    result = []
    for dataset_id, dataset in enumerate(datasets):
        for record in dataset:
            record_with_dataset_id = record[:id_col_index] + (dataset_id,) + record[id_col_index:]
            result.append(record_with_dataset_id)
    return result


def find_duplicates_across(datasets: list[list[tuple]], t: float) -> set[tuple[tuple, tuple, float]]:
    dataset_id_index = 0
    merged = _merge_datasets(datasets, dataset_id_index)
    raw_results = find_duplicates(merged, t, exclude_cols=[dataset_id_index])
    results = set()
    for dupe in raw_results:
        x, y, sim = dupe
        x_dataset_id = x[dataset_id_index]
        y_dataset_id = y[dataset_id_index]
        if x_dataset_id == y_dataset_id:
            continue
        results.add(dupe)
    return results
