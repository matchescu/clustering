from entity_matchers.ppjoin import find_duplicates
from entity_matchers.ppjoin._algo import find_duplicates_across, merge_duplicates, merge_duplicates_across


def test_deduplicate_sample_data_with_half_jaccard(sample_data):
    result = list(find_duplicates(sample_data, 0.5))

    assert len(result) == 1
    assert result[0][2] >= 0.5


def test_deduplicate_exact_matches(sample_data):
    result = list(find_duplicates(sample_data, 0.9))

    assert len(result) == 0


def test_find_duplicates_across_two_datasets(two_data_sets):
    result = list(find_duplicates_across(two_data_sets, 0.25))

    assert len(result) == 1
    row_a = result[0][0]
    row_b = result[0][1]
    sim = result[0][2]
    assert row_a[0] != row_b[0]
    assert sim >= 0.25


def test_merge_duplicates():
    a = ("a", "e", "f")
    b = ("e", "c")
    c = ("d", "f")

    result = merge_duplicates([a, b, c])

    assert result == ("a", "e", "c", "d", "f")


def test_merge_duplicates_across(two_data_sets):
    result = list(merge_duplicates_across(two_data_sets, 0.25))

    assert result == [(
        '304587',
        'A user-centered interface for querying distributed multimedia databases',
        'Isabel F. Cruz, Kimberly M. James',
        'International Conference on Management of Data',
        '1999'
    )]
