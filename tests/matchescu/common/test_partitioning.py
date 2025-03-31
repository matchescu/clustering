import pytest

from matchescu.clustering._ecp import EquivalenceClassPartitioner


@pytest.fixture
def compute_partition(comparison_space):
    return EquivalenceClassPartitioner(comparison_space)


@pytest.mark.parametrize(
    "comparison_space,similarity_graph", [([("a", "a")], [("a", "a")])], indirect=True
)
def test_reflexivity(compute_partition, comparison_space, similarity_graph, ref_id):
    partition = compute_partition(similarity_graph)
    assert len(partition) == 1
    assert partition == frozenset({frozenset({ref_id("a", "test")})})


@pytest.mark.parametrize(
    "comparison_space,similarity_graph",
    [([("a", "b")], [("a", "b"), ("b", "a")])],
    indirect=True,
)
def test_symmetry(compute_partition, similarity_graph, comparison_space, ref_id):
    partition = compute_partition(similarity_graph)
    assert len(partition) == 1
    assert partition == frozenset(
        {frozenset({ref_id("a", "test"), ref_id("b", "test")})}
    )


@pytest.mark.parametrize(
    "comparison_space,similarity_graph",
    [([("a", "b"), ("b", "c")], [("a", "b"), ("b", "c")])],
    indirect=True,
)
def test_transitivity(compute_partition, similarity_graph, comparison_space, ref_id):
    partition = compute_partition(similarity_graph)

    assert len(partition) == 1
    assert partition == frozenset(
        {frozenset({ref_id("a", "test"), ref_id("b", "test"), ref_id("c", "test")})}
    )


@pytest.mark.parametrize(
    "comparison_space,similarity_graph",
    [([("a", "b"), ("b", "c"), ("c", "d")], [("a", "b"), ("b", "c"), ("d", "a")])],
    indirect=True,
)
def test_create_single_set(
    compute_partition, similarity_graph, comparison_space, ref_id
):
    partition = compute_partition(similarity_graph)

    assert len(partition) == 1
    assert partition == frozenset(
        {
            frozenset(
                {
                    ref_id("a", "test"),
                    ref_id("b", "test"),
                    ref_id("c", "test"),
                    ref_id("d", "test"),
                }
            )
        }
    )


@pytest.mark.parametrize(
    "comparison_space,similarity_graph",
    [([("a", "b"), ("b", "c"), ("c", "d")], [("a", "b"), ("b", "c")])],
    indirect=True,
)
def test_isolated_item(compute_partition, similarity_graph, comparison_space, ref_id):
    partition = compute_partition(similarity_graph)

    assert len(partition) == 2
    assert partition == frozenset(
        {
            frozenset(
                {
                    ref_id("a", "test"),
                    ref_id("b", "test"),
                    ref_id("c", "test"),
                }
            ),
            frozenset({ref_id("d", "test")}),
        }
    )
