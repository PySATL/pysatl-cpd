import csv
import tempfile
from pathlib import Path

import pytest

from benchmarking.storages.loaders.csv_loader.csv_loader import LoaderCSV


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

        with open(test_dir / "literal_literal.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["0", "42"])

        with open(test_dir / "list_list.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["0", "first"])
            writer.writerow(["1", "second"])
            writer.writerow(["2", "3.14"])

        yield test_dir


def test_loader_csv_initialization():
    loader = LoaderCSV(step_storages_name="test_init")
    assert loader.directory == Path("experiment_storages/csv/test_init")


def test_loader_csv_correct_data(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"correct"})

    assert "correct" in result
    assert result["correct"] == {"temp1": 23.5, "temp2": 24.1}


def test_loader_csv_invalid_data(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"invalid"})

    assert "invalid" in result
    assert result["invalid"] == {"val1": 10.5, "val2": "not_a_float"}


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

    assert "missing" not in result


def test_loader_csv_multiple_files(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"correct", "invalid", "empty", "missing"})

    assert set(result.keys()) == {"correct", "invalid", "empty"}
    assert result["correct"] == {"temp1": 23.5, "temp2": 24.1}
    assert result["invalid"] == {"val1": 10.5, "val2": "not_a_float"}
    assert result["empty"] == {}


def test_loader_csv_literal_loading(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"literal"})

    assert "literal" in result
    FOURTYTWO = 42
    assert result["literal"] == FOURTYTWO


def test_loader_csv_list_loading(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"list"})

    assert "list" in result
    assert result["list"] == ["first", "second", 3.14]


def test_loader_csv_special_characters(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader({"special"})

    assert "special" not in result


def test_loader_csv_empty_keys(csv_test_files):
    loader = LoaderCSV(step_storages_name="test_step")
    loader.directory = csv_test_files

    result = loader(set())

    assert result == {}


def test_loader_csv_directory_not_exists():
    loader = LoaderCSV(step_storages_name="non_existent")

    result = loader({"any_key"})

    assert "any_key" not in result
