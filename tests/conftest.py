import itertools
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
def comparison_space(entity_reference_id_set):
    result = MagicMock(spec=BinaryComparisonSpace)
    result.__iter__.return_value = list(itertools.combinations(entity_reference_id_set, 2))
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
def similarity_graph(ref, matcher_mock, max_non_match_threshold, min_match_threshold, source):
    sim_graph = SimilarityGraph(matcher_mock, max_non_match_threshold, min_match_threshold)
    sim_graph.add(ref("a", source), ref("b", source))
    sim_graph.add(ref("b", source), ref("c", source))
    sim_graph.add(ref("c", source), ref("d", source))
    return sim_graph