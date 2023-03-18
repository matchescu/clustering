import csv
import pytest

from pathlib import Path


TEST_DIR = Path(__file__).parent


def _load_data(filename: str) -> list[tuple]:
    result = []
    with open(TEST_DIR / "data" / filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            result.append(tuple(row))
    return result


@pytest.fixture
def sample_data() -> list[tuple]:
    return _load_data("sample.csv")


@pytest.fixture
def two_data_sets() -> list[list[tuple], list[tuple]]:
    return [_load_data("subset_a.csv"), _load_data("subset_b.csv")]
