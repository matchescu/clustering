import itertools
from functools import reduce
from pathlib import Path
from typing import Hashable, Callable
from unittest.mock import MagicMock

import pytest

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.similarity import SimilarityGraph, Matcher
from matchescu.typing import EntityReferenceIdentifier, EntityReference


@pytest.fixture(scope="session")
def test_dir():
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(test_dir):
    return test_dir / "data"


@pytest.fixture
def source():
    return "test"


@pytest.fixture(scope="session")
def ref_id() -> Callable[[Hashable, str], EntityReferenceIdentifier]:
    return lambda lbl, src: EntityReferenceIdentifier(lbl, src)


@pytest.fixture(scope="session")
def ref(ref_id):
    def _(lbl, src):
        mock = MagicMock(name=f"{lbl}-reference", spec=EntityReference)
        mock.id = ref_id(lbl, src)
        return mock

    return _


@pytest.fixture
def entity_reference_id_set(ref_id, source) -> list[EntityReferenceIdentifier]:
    return [
        ref_id("a", source),
        ref_id("b", source),
        ref_id("c", source),
        ref_id("d", source),
    ]


@pytest.fixture
def comparison_space(entity_reference_id_set, request, ref_id, source):
    result = MagicMock(spec=BinaryComparisonSpace)
    cmp_space = list(itertools.combinations(entity_reference_id_set, 2))
    if hasattr(request, "param") and isinstance(request.param, list):
        cmp_space = [(ref_id(x, source), ref_id(y, source)) for x, y in request.param]
    result.__iter__.return_value = cmp_space
    return result


@pytest.fixture
def max_non_match_threshold(request):
    return request.param if hasattr(request, "param") else 0.25


@pytest.fixture
def min_match_threshold(request):
    return request.param if hasattr(request, "param") else 0.75


@pytest.fixture
def matcher_mock(request):
    result = MagicMock(name="mock matcher object", spec=Matcher)
    result.return_value = request.param if hasattr(request, "param") else 1.0
    return result


@pytest.fixture
def similarity_graph(
    ref, matcher_mock, max_non_match_threshold, min_match_threshold, source, request
):
    edge_spec = [
        (ref("a", source), ref("b", source)),
        (ref("b", source), ref("c", source)),
        (ref("c", source), ref("d", source)),
    ]
    if hasattr(request, "param") and isinstance(request.param, (list, set, tuple)):
        edge_spec = list((ref(x, source), ref(y, source)) for x, y in request.param)
    sim_graph = reduce(
        lambda g, pair: g.add(*pair),
        edge_spec,
        SimilarityGraph(matcher_mock, max_non_match_threshold, min_match_threshold),
    )
    return sim_graph
