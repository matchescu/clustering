import polars as pl
import pytest

from matchescu.matching.entity_reference import RawComparison, EntityReferenceComparisonConfig
from matchescu.matching.ml.datasets import CsvDataSource, Traits, DuplicationDataSet


@pytest.fixture
def dataset_dir(data_dir):
    return data_dir / "cora"


@pytest.fixture
def cora(dataset_dir):
    traits = list(Traits().int([0]).string([2, 3, 5, 7]))
    return CsvDataSource("cora", traits).read_csv(dataset_dir / "cora.csv")


@pytest.fixture
def comparison_config() -> EntityReferenceComparisonConfig:
    return (
        RawComparison().levenshtein("name", 2, 2)
    )


def test_cora(cora, comparison_config):
    n = len(cora)
    sut = DuplicationDataSet(cora, set()).attr_compare(comparison_config)

    sut.cross_sources()

    assert len(sut.feature_matrix) == (n*(n-1)) / 2