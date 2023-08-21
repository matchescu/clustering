from matchescu.entity_matchers import ppjoin_adapter


def test_happy_flow(subsample_a, subsample_b):
    ppjoin_adapter([subsample_a, subsample_b])

    assert True
