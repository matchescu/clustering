import polars as pl
import pytest

from matchescu.matching.entity_reference._comparison import FellegiSunterComparison
from matchescu.matching.ml.datasets import CsvDataSource, Traits, RecordLinkageDataSet


@pytest.fixture
def dataset_dir(data_dir):
    return data_dir / "abt-buy"


@pytest.fixture
def left_source(dataset_dir):
    traits = list(Traits().int([0]).string([1, 2]).currency([3]))
    return CsvDataSource("abt", traits).read_csv(dataset_dir / "Abt.csv")


@pytest.fixture
def right_source(dataset_dir):
    traits = list(Traits().int([0]).string([1, 2, 3]).currency([4]))
    return CsvDataSource("buy", traits).read_csv(dataset_dir / "Buy.csv")


@pytest.fixture
def true_matches(dataset_dir):
    perfect_mapping_path = dataset_dir / "abt_buy_perfectMapping.csv"
    return set(
        pl.read_csv(perfect_mapping_path, ignore_errors=True).iter_rows()
    )


@pytest.fixture
def record_linkage_dataset(left_source, right_source, true_matches):
    return RecordLinkageDataSet(left_source, right_source, true_matches)


@pytest.fixture
def comparison_config():
    return FellegiSunterComparison().jaccard(
        "name", 1, 1
    ).jaccard(
        "description", 2, 2
    ).exact(
        "price", 3, 4
    )


def test_target_vector(record_linkage_dataset, left_source, right_source, true_matches):
    expected_size = len(left_source) * len(right_source)

    result = record_linkage_dataset._compute_target_vector()

    assert len(result) == expected_size
    assert len(result[result == 1]) == len(true_matches)


def test_feature_matrix(record_linkage_dataset, left_source, right_source, comparison_config):
    expected_size = len(left_source) * len(right_source)
    result = record_linkage_dataset.compute_feature_matrix(comparison_config)

    assert result.shape == (expected_size, 4)
