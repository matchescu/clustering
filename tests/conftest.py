from pathlib import Path
from typing import Hashable, Callable

import pytest

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
def entity_reference_id_set(ref_id) -> list[EntityReferenceIdentifier]:
    return [
        ref_id("a", "test"),
        ref_id("b", "test"),
        ref_id("c", "test"),
        ref_id("d", "test"),
    ]
