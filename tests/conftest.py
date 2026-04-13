import csv
import os
from functools import reduce, partial
from pathlib import Path
from typing import Hashable, Callable
from unittest.mock import MagicMock

import pytest

from matchescu.clustering import EquivalenceClassPartitioner
from matchescu.similarity import (
    ReferenceGraph,
    GmlGraphPersistence,
    MatchResult,
)
from matchescu.typing import EntityReferenceIdentifier, EntityReference


@pytest.fixture(scope="session")
def test_dir():
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(test_dir):
    return test_dir / "data"


@pytest.fixture
def dataset(data_dir, request):
    default = "beer"
    if not hasattr(request, "param"):
        return default
    dataset_name = str(request.param)
    if not Path(data_dir / dataset_name).is_dir():
        raise AssertionError(f"{dataset_name} is not a dir")
    return dataset_name


@pytest.fixture
def file_digraph(data_dir, request):
    file_name = request.param if hasattr(request, "param") else "beer-fwd-digraph.gml"
    persistence = GmlGraphPersistence(data_dir / file_name)
    return persistence.load()


def _get_dataset_fpath(data_dir, dataset, file_name):
    fpath = data_dir / dataset / file_name
    if not os.access(fpath, os.R_OK):
        raise AssertionError(f"can't read from {fpath}")
    return fpath


@pytest.fixture
def dataset_refs(data_dir, dataset):
    fpath = _get_dataset_fpath(data_dir, dataset, "all-refs.csv")
    with open(fpath, "r") as f:
        reader = csv.reader(f.readlines())
        col_names = {name: idx for idx, name in enumerate(next(reader))}
        id_col = col_names["id"]
        source_col = col_names["source"]
        all_refs = [
            EntityReferenceIdentifier(int(row[id_col]), row[source_col])
            for row in reader
        ]
    return all_refs


@pytest.fixture
def dataset_ground_truth(data_dir, dataset, dataset_refs):
    fpath = _get_dataset_fpath(data_dir, dataset, "ground-truth.csv")
    ecp = EquivalenceClassPartitioner(dataset_refs)
    with open(fpath, "r") as f:
        reader = csv.reader(f.readlines())
        col_names = next(reader)
        pairs = [
            (
                EntityReferenceIdentifier(label=int(row[0]), source=col_names[0]),
                EntityReferenceIdentifier(label=int(row[1]), source=col_names[1]),
            )
            for row in reader
        ]
    ground_truth_clusters = ecp(pairs)
    return ground_truth_clusters


@pytest.fixture
def dataset_fwd_graph(matcher_mock, data_dir, dataset):
    fpath = _get_dataset_fpath(data_dir, dataset, "fwd-digraph.gml")
    persistence = GmlGraphPersistence(fpath)
    return ReferenceGraph().load(persistence)


@pytest.fixture
def dataset_rev_graph(matcher_mock, data_dir, dataset):
    fpath = _get_dataset_fpath(data_dir, dataset, "rev-digraph.gml")
    persistence = GmlGraphPersistence(fpath)
    return ReferenceGraph().load(persistence)


@pytest.fixture
def dataset_bidi_graph(dataset_fwd_graph, dataset_rev_graph):
    return dataset_fwd_graph.merge(dataset_rev_graph)


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


def default_scoring_algorithm(a, b):
    return MatchResult(a.id, b.id, 1, [0, 1])


@pytest.fixture
def matcher_mock(request):
    scoring_algo = default_scoring_algorithm

    if hasattr(request, "param"):
        if isinstance(request.param, (int, float)):

            def _return_param(a, b):
                score = request.param
                return MatchResult(a.id, b.id, score, [1 - score, score])

            scoring_algo = _return_param
        elif isinstance(request.param, dict):

            def _match_score(x, y, score_dict):
                score = score_dict.get((x.id.label, y.id.label), 0.0)
                return MatchResult(x.id, y.id, score, [1 - score, score])

            scoring_algo = partial(_match_score, score_dict=request.param)

    return MagicMock(name="mock matcher object", side_effect=scoring_algo)


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
        lambda g, pair: g.add(matcher_mock(*pair)),
        edge_spec,
        ReferenceGraph(directed),
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
