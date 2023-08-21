import csv

import pandas
import pytest

from pathlib import Path


TEST_DIR = Path(__file__).parent


def _load_data(filename: str, headers: bool) -> pandas.DataFrame:
    with open(TEST_DIR / "data" / filename, "r") as csv_file:
        return pandas.read_csv(csv_file, header=1 if headers else 0)


@pytest.fixture
def subsample_a():
    return _load_data("subsample_a.csv", False)


@pytest.fixture
def subsample_b():
    return _load_data("subsample_b.csv", False)
