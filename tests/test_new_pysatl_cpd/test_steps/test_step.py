import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.step import Step
from tests.test_new_pysatl_cpd.test_steps.test_data_generation_step.test_data_handlers.mock_data_handler import (
    MockDataHandler,
)
from tests.test_new_pysatl_cpd.test_steps.test_experiment_execution_step.test_workers.mock_worker import MockWorker
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_report_builders.mock_report_builder import (
    MockReportBuilder,
)

# TODO remove ruff exception
# ruff: noqa: E501
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_report_visualizers.mock_report_visualizer import (
    MockReportVisualizer,
)
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_reporters.mock_reporter import MockReporter


class TestStep:
    mock_data_handler = MockDataHandler()
    mock_data_generation_step = DataGenerationStep(mock_data_handler)
    mock_worker = MockWorker()
    mock_experiment_execution_step = ExperimentExecutionStep(mock_worker)
    mock_report_builder = MockReportBuilder()
    mock_report_visualizer = MockReportVisualizer()
    mock_reporter = MockReporter(mock_report_builder, mock_report_visualizer)
    mock_report_generation_step = ReportGenerationStep(mock_reporter)
    MAX_ITERATIONS = 200

    @pytest.mark.parametrize(
        "step_1,step_2,expected",
        [
            (mock_data_generation_step, mock_data_generation_step, True),
            (mock_data_generation_step, mock_experiment_execution_step, True),
            (mock_data_generation_step, mock_report_generation_step, False),
            (mock_experiment_execution_step, mock_data_generation_step, False),
            (mock_experiment_execution_step, mock_experiment_execution_step, True),
            (mock_experiment_execution_step, mock_report_generation_step, True),
            (mock_report_generation_step, mock_data_generation_step, False),
            (mock_report_generation_step, mock_experiment_execution_step, False),
            (mock_report_generation_step, mock_report_generation_step, True),
        ],
    )
    def test_set_next(self, step_1, step_2, expected):
        # TODO add test for exception value (err)
        try:
            step_1.set_next(step_2)
            assert expected
        except ValueError:
            assert not expected

    @settings(max_examples=MAX_ITERATIONS)
    @given(source_dict=st.dictionaries(st.text(), st.floats(allow_nan=False)), reference_set=st.sets(st.text()))
    def test_filter_and_rename_set(self, source_dict, reference_set):
        # TODO add test for exception value (err)
        step = Step

        if not reference_set:
            result = step._filter_and_rename(source_dict, reference_set)
            assert result == {}
            return

        if reference_set.issubset(source_dict.keys()):
            result = step._filter_and_rename(source_dict, reference_set)
            assert set(result.keys()) == reference_set
            for key in reference_set:
                if result[key] != source_dict[key]:
                    cpd_logger.debug(f"{result[key]} {source_dict[key]}")
                assert result[key] == source_dict[key]
            return

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        source_dict=st.dictionaries(st.text(), st.floats(allow_nan=False)),
        reference_dict=st.dictionaries(st.text(), st.text()),
    )
    def test_filter_and_rename_dict(self, source_dict, reference_dict):
        step = Step

        if len(set(reference_dict.values())) != len(reference_dict.values()):
            # TODO add test for exception value (err)
            assert True
            return

        if not reference_dict:
            result = step._filter_and_rename(source_dict, reference_dict)
            assert result == {}
            return

        if set(reference_dict.keys()).issubset(source_dict.keys()):
            result = step._filter_and_rename(source_dict, reference_dict)
            assert set(result.keys()) == set(reference_dict.values())
            for old_key, new_key in reference_dict.items():
                assert result[new_key] == source_dict[old_key]
            return

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        input_data=st.dictionaries(st.text(), st.floats(allow_nan=False)),
        input_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        input_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
    )
    def test_get_step_input(
        self, input_data, input_step_names, input_storage_names, output_step_names, output_storage_names
    ):
        step = random.choice(
            [self.mock_data_generation_step, self.mock_experiment_execution_step, self.mock_report_generation_step]
        )
        if isinstance(input_step_names, dict) and len(set(input_step_names.values())) != len(input_step_names.values()):
            # TODO add test for exception value (err)
            assert True
            return

        # Save original values
        original_input_step_names = step.input_step_names
        original_input_storage_names = step.input_storage_names
        original_output_step_names = step.output_step_names
        original_output_storage_names = step.output_storage_names

        try:
            # Set new values
            step.input_step_names = input_step_names
            step.input_storage_names = input_storage_names
            step.output_step_names = output_step_names
            step.output_storage_names = output_storage_names

            if not input_step_names:
                result = step._get_step_input(input_data)
                assert result == {}
                return

            if isinstance(input_step_names, set):
                if input_step_names.issubset(input_data.keys()):
                    result = step._get_step_input(input_data)
                    assert set(result.keys()) == input_step_names
                    for key in input_step_names:
                        assert result[key] == input_data[key]
                else:
                    # TODO add test for exception value (err)
                    assert True
            elif set(input_step_names.keys()).issubset(input_data.keys()):
                result = step._get_step_input(input_data)
                assert set(result.keys()) == set(input_step_names.values())
                for old_key, new_key in input_step_names.items():
                    assert result[new_key] == input_data[old_key]
            else:
                # TODO add test for exception value (err)
                assert True
        finally:
            # Restore original values
            step.input_step_names = original_input_step_names
            step.input_storage_names = original_input_storage_names
            step.output_step_names = original_output_step_names
            step.output_storage_names = original_output_storage_names

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        input_data=st.dictionaries(st.text(), st.floats(allow_nan=False)),
        input_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        input_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
    )
    def test_get_storage_input(
        self, input_data, input_step_names, input_storage_names, output_step_names, output_storage_names
    ):
        step = random.choice(
            [self.mock_data_generation_step, self.mock_experiment_execution_step, self.mock_report_generation_step]
        )
        if isinstance(input_storage_names, dict) and len(set(input_storage_names.values())) != len(
            input_storage_names.values()
        ):
            # TODO add test for exception value (err)
            assert True
            return

        # Save original values
        original_input_step_names = step.input_step_names
        original_input_storage_names = step.input_storage_names
        original_output_step_names = step.output_step_names
        original_output_storage_names = step.output_storage_names

        try:
            # Set new values
            step.input_step_names = input_step_names
            step.input_storage_names = input_storage_names
            step.output_step_names = output_step_names
            step.output_storage_names = output_storage_names

            if not input_storage_names:
                result = step._get_storage_input(input_data)
                assert result == {}
                return

            if isinstance(input_storage_names, set):
                if input_storage_names.issubset(input_data.keys()):
                    result = step._get_storage_input(input_data)
                    assert set(result.keys()) == input_storage_names
                    for key in input_storage_names:
                        assert result[key] == input_data[key]
                else:
                    # TODO add test for exception value (err)
                    assert True
            elif set(input_storage_names.keys()).issubset(input_data.keys()):
                result = step._get_storage_input(input_data)
                assert set(result.keys()) == set(input_storage_names.values())
                for old_key, new_key in input_storage_names.items():
                    assert result[new_key] == input_data[old_key]
            else:
                # TODO add test for exception value (err)
                assert True
        finally:
            # Restore original values
            step.input_step_names = original_input_step_names
            step.input_storage_names = original_input_storage_names
            step.output_step_names = original_output_step_names
            step.output_storage_names = original_output_storage_names

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        output_data=st.dictionaries(st.text(), st.floats(allow_nan=False)),
        input_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        input_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
    )
    def test_get_step_output(
        self, output_data, input_step_names, input_storage_names, output_step_names, output_storage_names
    ):
        step = random.choice(
            [self.mock_data_generation_step, self.mock_experiment_execution_step, self.mock_report_generation_step]
        )

        if isinstance(output_step_names, dict) and len(set(output_step_names.values())) != len(
            output_step_names.values()
        ):
            # TODO add test for exception value (err)
            assert True
            return

        # Save original values
        original_input_step_names = step.input_step_names
        original_input_storage_names = step.input_storage_names
        original_output_step_names = step.output_step_names
        original_output_storage_names = step.output_storage_names

        try:
            # Set new values
            step.input_step_names = input_step_names
            step.input_storage_names = input_storage_names
            step.output_step_names = output_step_names
            step.output_storage_names = output_storage_names

            if not output_step_names:
                result = step._get_step_output(output_data)
                assert result == {}
                return

            if isinstance(output_step_names, set):
                if output_step_names.issubset(output_data.keys()):
                    result = step._get_step_output(output_data)
                    assert set(result.keys()) == output_step_names
                    for key in output_step_names:
                        assert result[key] == output_data[key]
                else:
                    # TODO add test for exception value (err)
                    assert True
            elif set(output_step_names.keys()).issubset(output_data.keys()):
                result = step._get_step_output(output_data)
                assert set(result.keys()) == set(output_step_names.values())
                for old_key, new_key in output_step_names.items():
                    assert result[new_key] == output_data[old_key]
            else:
                # TODO add test for exception value (err)
                assert True
        finally:
            # Restore original values
            step.input_step_names = original_input_step_names
            step.input_storage_names = original_input_storage_names
            step.output_step_names = original_output_step_names
            step.output_storage_names = original_output_storage_names

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        output_data=st.dictionaries(st.text(), st.floats(allow_nan=False)),
        input_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        input_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
    )
    def test_get_storage_output(
        self, output_data, input_step_names, input_storage_names, output_step_names, output_storage_names
    ):
        step = random.choice(
            [self.mock_data_generation_step, self.mock_experiment_execution_step, self.mock_report_generation_step]
        )

        if isinstance(output_storage_names, dict) and len(set(output_storage_names.values())) != len(
            output_storage_names.values()
        ):
            # TODO add test for exception value (err)
            assert True
            return

        # Save original values
        original_input_step_names = step.input_step_names
        original_input_storage_names = step.input_storage_names
        original_output_step_names = step.output_step_names
        original_output_storage_names = step.output_storage_names

        try:
            # Set new values
            step.input_step_names = input_step_names
            step.input_storage_names = input_storage_names
            step.output_step_names = output_step_names
            step.output_storage_names = output_storage_names

            if not output_storage_names:
                result = step._get_storage_output(output_data)
                assert result == {}
                return

            if isinstance(output_storage_names, set):
                if output_storage_names.issubset(output_data.keys()):
                    result = step._get_storage_output(output_data)
                    assert set(result.keys()) == output_storage_names
                    for key in output_storage_names:
                        assert result[key] == output_data[key]
                else:
                    # TODO add test for exception value (err)
                    assert True
            elif set(output_storage_names.keys()).issubset(output_data.keys()):
                result = step._get_storage_output(output_data)
                assert set(result.keys()) == set(output_storage_names.values())
                for old_key, new_key in output_storage_names.items():
                    assert result[new_key] == output_data[old_key]
            else:
                # TODO add test for exception value (err)
                assert True
        finally:
            # Restore original values
            step.input_step_names = original_input_step_names
            step.input_storage_names = original_input_storage_names
            step.output_step_names = original_output_step_names
            step.output_storage_names = original_output_storage_names

    @settings(max_examples=MAX_ITERATIONS)
    @given(
        input_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        input_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_step_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
        output_storage_names=st.one_of(st.sets(st.text()), st.dictionaries(st.text(), st.text())),
    )
    def test_set_storage_data_from_processor(
        self, input_step_names, input_storage_names, output_step_names, output_storage_names
    ):
        step = random.choice(
            [self.mock_data_generation_step, self.mock_experiment_execution_step, self.mock_report_generation_step]
        )

        # Save original values
        original_input_step_names = step.input_step_names
        original_input_storage_names = step.input_storage_names
        original_output_step_names = step.output_step_names
        original_output_storage_names = step.output_storage_names
        original_fields_info_none_mask = step._fields_info_none_mask.copy()

        try:
            # Set new values
            step.input_step_names = input_step_names
            step.input_storage_names = input_storage_names
            step.output_step_names = output_step_names
            step.output_storage_names = output_storage_names

            # Update fields_info_none_mask based on new values
            step._fields_info_none_mask = {
                "input_storage_names": input_storage_names is None,
                "output_storage_names": output_storage_names is None,
                "input_step_names": input_step_names is None,
                "output_step_names": output_step_names is None,
            }

            # Get processor based on step type
            if isinstance(step, DataGenerationStep):
                processor = step.data_handler
            elif isinstance(step, ExperimentExecutionStep):
                processor = step._worker
            else:  # ReportGenerationStep
                processor = step._reporter

            step._set_storage_data_from_processor(processor)

            if step._fields_info_none_mask["input_storage_names"]:
                assert step.input_storage_names == processor.input_storage_names
            if step._fields_info_none_mask["output_storage_names"]:
                assert step.output_storage_names == processor.output_storage_names
            if step._fields_info_none_mask["input_step_names"]:
                assert step.input_step_names == processor.input_step_names
            if step._fields_info_none_mask["output_step_names"]:
                assert step.output_step_names == processor.output_step_names

        finally:
            # Restore original values
            step.input_step_names = original_input_step_names
            step.input_storage_names = original_input_storage_names
            step.output_step_names = original_output_step_names
            step.output_storage_names = original_output_storage_names
            step._fields_info_none_mask = original_fields_info_none_mask
