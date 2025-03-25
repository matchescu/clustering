import pytest

from matchescu.clustering._ecp import EquivalenceClassPartitioner


@pytest.fixture
def compute_partition(entity_reference_id_set, request):
    items = (
        request.param
        if hasattr(request, "param") and isinstance(request.param, list)
        else entity_reference_id_set
    )
    return EquivalenceClassPartitioner(items)


@pytest.mark.parametrize("compute_partition", [["a"]], indirect=True)
def test_reflexivity(compute_partition):
    partition = compute_partition([("a", "a")])
    assert len(partition) == 1
    assert partition == [["a"]]


@pytest.mark.parametrize("compute_partition", [["a", "b"]], indirect=True)
def test_symmetry(compute_partition):
    partition = compute_partition([("a", "b"), ("b", "a")])
    assert len(partition) == 1
    assert partition == [["a", "b"]]


@pytest.mark.parametrize("compute_partition", [["a", "b", "c"]], indirect=True)
def test_transitivity(compute_partition):
    partition = compute_partition([("a", "b"), ("b", "c")])

    assert len(partition) == 1
    assert partition == [["a", "b", "c"]]


@pytest.mark.parametrize("compute_partition", [["a", "b", "c", "d"]], indirect=True)
def test_create_single_set(compute_partition):
    partition = compute_partition([("a", "b"), ("b", "c"), ("d", "a")])

    assert len(partition) == 1
    assert partition == [["a", "b", "c", "d"]]


@pytest.mark.parametrize("compute_partition", [["a", "b", "c", "d"]], indirect=True)
def test_isolated_item(compute_partition):
    partition = compute_partition([("a", "b"), ("b", "c")])

    assert len(partition) == 2
    assert partition == [["a", "b", "c"], ["d"]]
