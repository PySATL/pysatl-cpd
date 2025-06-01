import csv
import tempfile
from pathlib import Path

import pytest

from new_pysatl_cpd.storages.loaders.csv_loader.csv_loader import LoaderCSV


@pytest.fixture
def csv_test_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "experiment_storages" / "csv" / "test_step"
        test_dir.mkdir(parents=True)

        with open(test_dir / "correct.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["temp1", "23.5"])
            writer.writerow(["temp2", "24.1"])

        with open(test_dir / "invalid.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["val1", "10.5"])
            writer.writerow(["val2", "not_a_float"])
            writer.writerow(["val3"])

        with open(test_dir / "empty.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])

        yield test_dir


def test_loader_csv_initialization():
    loader = LoaderCSV(step_storages_name="test_init")
    assert loader.directory == Path("experiment_storages/csv/test_init")


def test_loader_csv_correct_data(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"correct"})

    assert "correct" in result
    assert result["correct"] == {1: 23.5, 2: 24.1}


def test_loader_csv_invalid_data(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"invalid"})

    assert "invalid" in result
    assert result["invalid"] == {1: 10.5}


def test_loader_csv_empty_file(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"empty"})

    assert "empty" in result
    assert result["empty"] == {}


def test_loader_csv_missing_file(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"missing"})

    assert "missing" in result
    assert result["missing"] == {}


def test_loader_csv_multiple_files(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"correct", "invalid", "empty", "missing"})

    assert set(result.keys()) == {"correct", "invalid", "empty", "missing"}
    assert result["correct"] == {1: 23.5, 2: 24.1}
    assert result["invalid"] == {1: 10.5}
    assert result["empty"] == {}
    assert result["missing"] == {}


def test_loader_csv_special_characters():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "experiment_storages" / "csv" / "special"
        test_dir.mkdir(parents=True)

        with open(test_dir / "special.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["temp/℃", "25.5"])
            writer.writerow(["pres$ure", "1013.2"])

        loader = LoaderCSV(step_storages_name="special")
        loader.directory = test_dir

        result = loader({"special"})
        assert result["special"] == {1: 25.5, 2: 1013.2}


def test_loader_csv_empty_keys(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader(set())

    assert result == {}


def test_loader_csv_directory_not_exists():
    loader = LoaderCSV(step_storages_name="non_existent")

    result = loader({"any_key"})

    assert "any_key" in result
    assert result["any_key"] == {}
