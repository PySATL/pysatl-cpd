import csv
import tempfile
from pathlib import Path

import pytest

from benchmarking.storages.savers.csv_saver.csv_saver import SaverCSV


@pytest.fixture
def temp_experiment_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_saver_csv_initialization(temp_experiment_dir):
    step_name = "test_step"
    SaverCSV(step_storages_name=step_name)

    expected_dir = Path("experiment_storages") / "csv" / step_name
    assert expected_dir.exists()
    assert expected_dir.is_dir()


def test_saver_csv_single_file_creation(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="test_step")
    test_data = {"temperature": 23.5, "pressure": 1013.2}

    storage_name = "sensor_data"
    saver(storage_name, test_data)

    file_path = saver.directory / f"{storage_name}.csv"
    assert file_path.exists()
    assert file_path.is_file()


def test_saver_csv_file_content(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="content_test")
    test_data = {"value1": 10.5, "value2": 20.3}

    saver("test_data", test_data)
    file_path = saver.directory / "test_data.csv"

    with open(file_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["key", "value"]

        rows = list(reader)

        rows_count = 2
        assert len(rows) == rows_count
        assert {"value1", "value2"} == {row[0] for row in rows}
        assert {"10.5", "20.3"} == {row[1] for row in rows}


def test_saver_csv_multiple_files(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="multi_file_test")

    saver("day1", {"temp": 22.0})
    saver("day2", {"temp": 23.0})

    assert (saver.directory / "day1.csv").exists()
    assert (saver.directory / "day2.csv").exists()


def test_saver_csv_overwrite_existing(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="overwrite_test")

    saver("test", {"a": 1.0})

    saver("test", {"b": 2.0})

    file_path = saver.directory / "test.csv"
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == ["b", "2.0"]


def test_saver_csv_empty_data(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="empty_test")

    saver("empty_test", {})

    file_path = saver.directory / "empty_test.csv"
    assert file_path.exists()

    with open(file_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["key", "value"]
        assert len(list(reader)) == 0


def test_saver_csv_nested_directory(temp_experiment_dir):
    nested_name = "nested/test/path"
    saver = SaverCSV(step_storages_name=nested_name)

    assert saver.directory.exists()
    assert saver.directory.is_dir()

    saver("nested_test", {"x": 10.0})
    assert (saver.directory / "nested_test.csv").exists()


def test_saver_csv_special_characters(temp_experiment_dir):
    saver = SaverCSV(step_storages_name="special_chars")

    test_data = {"temp": 25.5, "pressure": 1013.2, "num1": 42.0}

    saver("special_data", test_data)

    file_path = saver.directory / "special_data.csv"
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader)
        rows = {row[0]: float(row[1]) for row in reader}

    assert rows == test_data
