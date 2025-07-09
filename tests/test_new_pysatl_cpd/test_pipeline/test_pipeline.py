import pytest

from benchmarking.pipeline.pipeline import Pipeline
from benchmarking.steps.data_generation_step.data_generation_step import DataGenerationStep
from benchmarking.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from benchmarking.steps.report_generation_step.report_generation_step import ReportGenerationStep
from benchmarking.steps.step import Step
from benchmarking.storages.loaders.default_loader import DefaultLoader
from benchmarking.storages.savers.default_saver import DefaultSaver
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


class TestPipeline:
    mock_data_handler = MockDataHandler()
    mock_data_generation_step = DataGenerationStep(mock_data_handler)
    mock_worker = MockWorker()
    mock_experiment_execution_step = ExperimentExecutionStep(mock_worker)
    mock_report_builder = MockReportBuilder()
    mock_report_visualizer = MockReportVisualizer()
    mock_reporter = MockReporter(mock_report_builder, mock_report_visualizer)
    mock_report_generation_step = ReportGenerationStep(mock_reporter)
    mock_saver_gen_data = DefaultSaver(dict())
    mock_loader_gen_data = DefaultLoader(dict())
    mock_saver_result = DefaultSaver(dict())
    mock_loader_result = DefaultLoader(dict())
    MAX_ITERATIONS = 200

    def test_setup_step_storage(self):
        pipeline = Pipeline([])
        pipeline._generated_data_saver = self.mock_saver_gen_data
        pipeline._result_saver = self.mock_saver_result
        pipeline._generated_data_loader = self.mock_loader_gen_data
        pipeline._result_loader = self.mock_loader_result

        self.mock_data_generation_step.saver = None
        self.mock_data_generation_step.loader = None
        pipeline._setup_step_storage(self.mock_data_generation_step)
        assert self.mock_data_generation_step.saver is self.mock_saver_gen_data
        assert self.mock_data_generation_step.loader is None

        self.mock_experiment_execution_step.saver = None
        self.mock_experiment_execution_step.loader = None
        pipeline._setup_step_storage(self.mock_experiment_execution_step)
        assert self.mock_experiment_execution_step.saver is self.mock_saver_result
        assert self.mock_experiment_execution_step.loader is self.mock_loader_gen_data

        self.mock_report_generation_step.saver = None
        self.mock_report_generation_step.loader = None
        pipeline._setup_step_storage(self.mock_report_generation_step)
        assert self.mock_report_generation_step.saver is None
        assert self.mock_report_generation_step.loader is self.mock_loader_result

    def test_check_two_steps(self):
        pipeline = Pipeline([])

        # Test 1: Valid DataGenerationStep -> ExperimentExecutionStep
        step1 = DataGenerationStep(
            self.mock_data_handler, output_storage_names={"data1", "data2"}, output_step_names={"meta1", "meta2"}
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker, input_storage_names={"data1", "data2"}, input_step_names={"meta1", "meta2"}
        )
        pipeline._check_two_steps(step1, step2)  # Should not raise any exception

        # Test 2: Valid ExperimentExecutionStep -> ReportGenerationStep
        step1 = ExperimentExecutionStep(
            self.mock_worker, output_storage_names={"result1", "result2"}, output_step_names={"meta3", "meta4"}
        )
        step2 = ReportGenerationStep(
            self.mock_reporter, input_storage_names={"result1", "result2"}, input_step_names={"meta3", "meta4"}
        )
        pipeline._check_two_steps(step1, step2)  # Should not raise any exception

        # Test 3: Invalid step type
        class InvalidStep(Step):
            def process(self, *args, **kwargs):
                return {}

            def _validate_storages(self):
                return True

        invalid_step = InvalidStep()
        with pytest.raises(ValueError, match="is unexpected Step"):
            pipeline._check_two_steps(invalid_step, step2)

        # Test 4: Missing storage fields
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_storage_names={"data1"},  # Only provides data1
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_storage_names={"data1", "data2"},  # Requires both data1 and data2
        )
        pipeline = Pipeline([])
        with pytest.raises(KeyError, match="must be values {'data2'} in the storage"):
            pipeline._check_two_steps(step1, step2)

        # Test 5: Missing step metadata
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_step_names={"meta1"},  # Only provides meta1
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_step_names={"meta1", "meta2"},  # Requires both meta1 and meta2
        )
        pipeline = Pipeline([])
        with pytest.raises(KeyError, match="must be values {'meta2'} returned from previous steps"):
            pipeline._check_two_steps(step1, step2)

        # Test 6: Dictionary-based field mapping
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_storage_names={"old_name": "new_name"},
            output_step_names={"old_meta": "new_meta"},
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker, input_storage_names={"new_name"}, input_step_names={"new_meta"}
        )
        pipeline._check_two_steps(step1, step2)  # Should not raise any exception

        # Test 7: Empty field sets
        step1 = DataGenerationStep(self.mock_data_handler, output_storage_names=set(), output_step_names=set())
        step2 = ExperimentExecutionStep(self.mock_worker, input_storage_names=set(), input_step_names=set())
        pipeline._check_two_steps(step1, step2)  # Should not raise any exception

    def test_config_pipeline(self):
        # Test 1: Valid pipeline configuration with all three step types
        step1 = DataGenerationStep(
            self.mock_data_handler, output_storage_names={"data1", "data2"}, output_step_names={"meta1", "meta2"}
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_storage_names={"data1", "data2"},
            input_step_names={"meta1", "meta2"},
            output_storage_names={"result1", "result2"},
            output_step_names={"meta3", "meta4"},
        )
        step3 = ReportGenerationStep(
            self.mock_reporter, input_storage_names={"result1", "result2"}, input_step_names={"meta3", "meta4"}
        )
        pipeline = Pipeline([step1, step2, step3])
        pipeline.config_pipeline()  # Should not raise any exception

        # Verify storage setup
        assert step1.saver is not None
        assert step1.loader is None
        assert step2.saver is not None
        assert step2.loader is not None
        assert step3.saver is None
        assert step3.loader is not None

        # Test 2: Empty pipeline
        pipeline = Pipeline([])
        pipeline.config_pipeline()  # Should not raise any exception

        # Test 3: Single step pipeline
        pipeline = Pipeline([step1])
        pipeline.config_pipeline()  # Should not raise any exception

        # Test 4: Invalid step type
        class InvalidStep(Step):
            def process(self, *args, **kwargs):
                return {}

            def _validate_storages(self):
                return True

        invalid_step = InvalidStep()
        with pytest.raises(ValueError, match="Unexpected type of"):
            pipeline = Pipeline([invalid_step])
            pipeline.config_pipeline()

        # Test 5: Incompatible step sequence
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_storage_names={"data1"},  # Only provides data1
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_storage_names={"data1", "data2"},  # Requires both data1 and data2
        )
        with pytest.raises(KeyError, match="must be values {'data2'} in the storage"):
            pipeline = Pipeline([step1, step2])
            pipeline.config_pipeline()

        # Test 6: Missing metadata
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_step_names={"meta1"},  # Only provides meta1
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_step_names={"meta1", "meta2"},  # Requires both meta1 and meta2
        )
        with pytest.raises(KeyError, match="must be values {'meta2'} returned from previous steps"):
            pipeline = Pipeline([step1, step2])
            pipeline.config_pipeline()

        # Test 7: Dictionary-based field mapping
        step1 = DataGenerationStep(
            self.mock_data_handler,
            output_storage_names={"old_name": "new_name"},
            output_step_names={"old_meta": "new_meta"},
        )
        step2 = ExperimentExecutionStep(
            self.mock_worker, input_storage_names={"new_name"}, input_step_names={"new_meta"}
        )
        pipeline = Pipeline([step1, step2])
        pipeline.config_pipeline()  # Should not raise any exception

        # Test 8: Multiple data generation steps
        step1 = DataGenerationStep(self.mock_data_handler, output_storage_names={"data1"}, output_step_names={"meta1"})
        step2 = DataGenerationStep(
            self.mock_data_handler,
            input_storage_names={"data1"},
            input_step_names={"meta1"},
            output_storage_names={"data2"},
            output_step_names={"meta2"},
        )
        step3 = ExperimentExecutionStep(self.mock_worker, input_storage_names={"data2"}, input_step_names={"meta2"})
        pipeline = Pipeline([step1, step2, step3])
        pipeline.config_pipeline()  # Should not raise any exception

        # Test 9: Multiple experiment execution steps
        step1 = DataGenerationStep(self.mock_data_handler, output_storage_names={"data1"}, output_step_names={"meta1"})
        step2 = ExperimentExecutionStep(
            self.mock_worker,
            input_storage_names={"data1"},
            input_step_names={"meta1"},
            output_storage_names={"result1"},
            output_step_names={"meta2"},
        )
        step3 = ExperimentExecutionStep(
            self.mock_worker,
            input_storage_names={"result1"},
            input_step_names={"meta2"},
            output_storage_names={"result2"},
            output_step_names={"meta3"},
        )
        step4 = ReportGenerationStep(self.mock_reporter, input_storage_names={"result2"}, input_step_names={"meta3"})
        pipeline = Pipeline([step1, step2, step3, step4])
        pipeline.config_pipeline()  # Should not raise any exception

    def test_run(self):
        # TODO test (hard to do because of saving data into storage.)
        assert True
