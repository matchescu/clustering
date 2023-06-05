from entity_matchers.ppjoin import find_duplicates, find_duplicates_across


def test_deduplicate_sample_data(sample_data):
    result = list(find_duplicates(sample_data, 0.66))

    assert len(result) == 1
    for item in result:
        assert item[-1] >= 0.5


def test_deduplicate_exact_matches(sample_data):
    result = list(find_duplicates(sample_data, 0.9))

    assert len(result) == 0


def test_find_duplicates_across_two_datasets(two_data_sets):
    result = list(find_duplicates_across(two_data_sets, 0.4))

    assert 0 < len(result) <= min(map(len, two_data_sets))
