from functools import reduce, partial
from pathlib import Path
from typing import Hashable, Callable
from unittest.mock import MagicMock

import pytest

from matchescu.similarity import ReferenceGraph, Matcher
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
def max_non_match_threshold(request):
    return request.param if hasattr(request, "param") else 0.25


@pytest.fixture
def min_match_threshold(request):
    return request.param if hasattr(request, "param") else 0.75


@pytest.fixture
def directed(request):
    return (not hasattr(request, "param")) or bool(request.param)


def default_scoring_algorithm(_, __):
    return 1.0


@pytest.fixture
def matcher_mock(request):
    scoring_algo = default_scoring_algorithm

    if hasattr(request, "param"):
        if isinstance(request.param, (int, float)):

            def _return_param(_, __):
                return request.param

            scoring_algo = _return_param
        elif isinstance(request.param, dict):

            def _match_score(x, y, score_dict):
                return score_dict.get((x.id.label, y.id.label), 0.0)

            scoring_algo = partial(_match_score, score_dict=request.param)

    return MagicMock(name="mock matcher object", spec=Matcher, side_effect=scoring_algo)


@pytest.fixture
def reference_graph(ref, matcher_mock, directed, source, request):
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
        ReferenceGraph(matcher_mock, directed),
    )
    return sim_graph


@pytest.fixture
def all_refs(request, ref_id, source):
    def _expand_item(item):
        if isinstance(item, tuple):
            yield from (ref_id(x, source) for x in item)
        else:
            yield ref_id(item, source)

    if hasattr(request, "param"):
        if isinstance(request.param, (list, set)):
            return {x for item in request.param for x in _expand_item(item)}
        if isinstance(request.param, dict):
            return {x for key in request.param for x in _expand_item(key)}

    return [
        ref_id("a", source),
        ref_id("b", source),
        ref_id("c", source),
        ref_id("d", source),
    ]
