from pathlib import Path
from typing import Hashable, Callable
from unittest.mock import MagicMock

import pytest

from matchescu.reference_store.comparison_space import BinaryComparisonSpace
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture(scope="session")
def test_dir():
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(test_dir):
    return test_dir / "data"


@pytest.fixture(scope="session")
def ref_id() -> Callable[[Hashable, str], EntityReferenceIdentifier]:
    return lambda lbl, src: EntityReferenceIdentifier(lbl, src)


@pytest.fixture
def comparison_space(ref_id, request):
    comparison_space = MagicMock(name="ComparisonSpaceMock", spec=BinaryComparisonSpace)
    result = [
        (ref_id("a", "test"), ref_id("b", "test")),
        (ref_id("b", "test"), ref_id("c", "test")),
        (ref_id("c", "test"), ref_id("d", "test")),
    ]
    if hasattr(request, "param") and isinstance(request.param, list):
        result = request.param
    comparison_space.__iter__.return_value = result
    return comparison_space
