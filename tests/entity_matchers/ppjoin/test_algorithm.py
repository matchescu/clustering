import pytest

from matchescu.entity_matchers import ppjoin_adapter


def test_fsm_is_list_of_pairs(sub_table_a, sub_table_b):
    result = ppjoin_adapter([sub_table_a, sub_table_b], threshold=0.2)

    for pair in result.fsm:
        assert len(pair) == 2


@pytest.mark.parametrize(
    "jaccard_threshold", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
def test_algebraic_is_partition(sub_table_a, sub_table_b, jaccard_threshold):
    result = ppjoin_adapter([sub_table_a, sub_table_b], threshold=jaccard_threshold)
    unique_a = set(row for row in sub_table_a.itertuples(index=False, name=None))
    unique_b = set(row for row in sub_table_b.itertuples(index=False, name=None))
    initial_set = unique_a | unique_b

    resulting_set = set(item for pair in result.algebraic for item in pair)
    resulting_subsets = list(
        set(item for item in subset) for subset in result.algebraic
    )

    for i, subset in enumerate(resulting_subsets):
        for j in range(i + 1, len(resulting_subsets)):
            assert len(subset & resulting_subsets[j]) == 0
    assert len(resulting_set & initial_set) == len(initial_set)
