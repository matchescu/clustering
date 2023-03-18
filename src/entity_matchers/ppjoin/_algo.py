from collections import defaultdict
from math import ceil
from typing import Any, Callable


def compute_prefix_length(x: tuple, t: float) -> int:
    return int(len(x) - ceil(t * len(x)) + 1)


def compute_alpha(x: tuple, y: tuple, t: float) -> float:
    coefficient = t / (1 + t)
    union_norm = len(x) + len(y)
    return ceil(coefficient * union_norm)


def _default_sort_key(token: Any) -> Any:
    return str(token)


def _to_multiset(matrix: list[tuple], sort_key: Callable[[Any], Any] = None) -> tuple[dict[int, int], list[tuple]]:
    sort_key = sort_key or _default_sort_key
    sorted_rows = list(
        (index, tuple(sorted((token for token in row), key=sort_key)))
        for index, row in enumerate(matrix)
    )

    def _by_len(x):
        return len(x[1])

    intermediate = sorted(sorted_rows, key=_by_len)
    row_index_mapping = {}
    multiset = []
    for sorted_index, item in enumerate(intermediate):
        original_index = item[0]
        sorted_row = item[1]
        multiset.append(sorted_row)
        row_index_mapping[sorted_index] = original_index
    return row_index_mapping, multiset


def _verify_candidates(
    data: list[tuple],
    multiset: list[tuple],
    row_index_mapping: dict[int, int],
    x_row_index,
    candidate_counts: dict[int, int],
    t: float,
    sort_key: Callable[[Any], Any] = None
) -> set[tuple[tuple, tuple, float]]:
    sort_key = sort_key or _default_sort_key
    result = set()
    for y_row_index in candidate_counts:
        common_token_count = candidate_counts[y_row_index]
        if common_token_count == 0:
            continue
        x = multiset[x_row_index]
        y = multiset[y_row_index]

        px = compute_prefix_length(x, t)
        py = compute_prefix_length(y, t)
        alpha = compute_alpha(x, y, t)

        overlap = common_token_count
        if sort_key(x[px-1]) < sort_key(y[py-1]):
            ubound = common_token_count + len(x) - px
            if ubound >= alpha:
                same_token_count = len(
                    set(x[px:]).intersection(set(y[common_token_count:]))
                )
                overlap += same_token_count
        else:
            ubound = common_token_count + len(y) - py
            if ubound >= alpha:
                same_token_count = len(
                    set(x[common_token_count:]).intersection(set(y[py:]))
                )
                overlap += same_token_count
        if overlap >= alpha:
            x_set = set(x)
            y_set = set(y)
            intersect_len = len(x_set.intersection(y_set))
            union_len = len(x_set.union(y_set))
            original_x = data[row_index_mapping[x_row_index]]
            original_y = data[row_index_mapping[y_row_index]]
            result.add((original_x, original_y, intersect_len/union_len))
    return result


def find_duplicates(data: list[tuple], t: float, sort_key: Callable = None) -> set[tuple[tuple, tuple, float]]:
    """Compute the similarity between the rows of a multiset."""
    if t < 0 or 1 <= t:
        raise ValueError(
            f"similarity threshold '{t}' must be between [0, 1)."
        )
    row_index_mapping, multiset = _to_multiset(data, sort_key)
    inverted_word_index = defaultdict(list[tuple[int, int]])
    result = set()
    for row_index, x in enumerate(multiset):
        # how many possible matches are there for a given record
        common_token_map = defaultdict(int)
        p = compute_prefix_length(x, t)
        for i in range(p):
            w = x[i]
            for y_index, j in inverted_word_index[w]:
                y = multiset[y_index]
                alpha = compute_alpha(x, y, t)
                min_suffix_len = min(len(x) - i, len(y) - j)
                ubound = min_suffix_len + 1
                if common_token_map.get(y_index, 0) + ubound >= alpha:
                    common_token_map[y_index] += 1
                else:
                    common_token_map[y_index] = 0
            inverted_word_index[w].append((row_index, i))
        partial_solutions = _verify_candidates(
            data, multiset, row_index_mapping, row_index, common_token_map, t, sort_key
        )
        result = result.union(partial_solutions)
    return result


def _merge_datasets(datasets: list[list[tuple]]) -> list[tuple]:
    result = []
    for dataset_id, dataset in enumerate(datasets):
        for record in dataset:
            result.append((dataset_id, *record))
    return result


def find_duplicates_across(datasets: list[list[tuple]], t: float) -> set[tuple[tuple, tuple, float]]:
    merged = _merge_datasets(datasets)
    raw_results = find_duplicates(merged, t)
    results = set()
    for dupe in raw_results:
        x, y, sim = dupe
        x_dataset_id = x[0]
        y_dataset_id = y[0]
        if x_dataset_id == y_dataset_id:
            continue
        results.add(dupe)
    return results


def merge_duplicates(dupes: list[tuple]) -> tuple:
    existing = dict()
    result = []
    union = [token for dupe in dupes for token in dupe]
    pos = 0
    for token in union:
        result.append(token)
        if token in existing:
            prev_pos = existing[token]
            del result[prev_pos]
            for idx in range(prev_pos, len(result)):
                existing[result[idx]] = idx
            pos = len(result) - 1
        else:
            existing[token] = pos
        pos += 1
    return tuple(result)


def merge_duplicates_across(datasets: list[list[tuple]], t: float) -> list[tuple]:
    duplicates = find_duplicates_across(datasets, t)
    for 