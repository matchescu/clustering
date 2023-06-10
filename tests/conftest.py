import csv
import pytest

from pathlib import Path

from abstractions.data_structures import Table

TEST_DIR = Path(__file__).parent


def _load_data(filename: str) -> list[list]:
    result = []
    with open(TEST_DIR / "data" / filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            result.append(row)
    return result


@pytest.fixture
def sample_data() -> list[list]:
    return _load_data("sample.csv")


@pytest.fixture
def two_data_sets() -> list[list[list], list[list]]:
    return [_load_data("subsample_a.csv"), _load_data("subsample_b.csv")]
