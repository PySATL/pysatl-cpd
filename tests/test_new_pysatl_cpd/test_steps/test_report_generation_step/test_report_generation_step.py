import pytest
from hypothesis import given
from hypothesis import strategies as st

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_builders.dummy_report_builder import DummyReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.dummy_report_visualizer import DummyReportVisualizer
from new_pysatl_cpd.steps.report_generation_step.reporters.dummy_reporter import DummyReporter
from new_pysatl_cpd.storages.loaders.default_loader import DefaultLoader
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_report_builders.mock_report_builder import (
    MockReportBuilder,
)

# TODO remove ruff exception
# ruff: noqa: E501
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_report_visualizers.mock_report_visualizer import (
    MockReportVisualizer,
)
from tests.test_new_pysatl_cpd.test_steps.test_report_generation_step.test_reporters.mock_reporter import MockReporter


class TestReportGenerationStep:
    @given(
        input_storage_names=st.sets(st.text()),
        output_storage_names=st.sets(st.text()),
        input_step_names=st.sets(st.text()),
        output_step_names=st.sets(st.text()),
    )
    def test_process_and_storage(self, input_storage_names, output_storage_names, input_step_names, output_step_names):
        # Setup
        builder = MockReportBuilder()
        visualizer = MockReportVisualizer()
        reporter = MockReporter(
            report_builder=builder,
            report_visualizer=visualizer,
            result={k: 1.00 for k in output_storage_names.union(output_step_names)},
        )

        if input_step_names.intersection(input_storage_names) or output_step_names.intersection(output_storage_names):
            with pytest.raises(ValueError):
                step = ReportGenerationStep(
                    reporter=reporter,
                    input_storage_names=input_storage_names,
                    output_storage_names=output_storage_names,
                    input_step_names=input_step_names,
                    output_step_names=output_step_names,
                )
            return

        step = ReportGenerationStep(
            reporter=reporter,
            input_storage_names=input_storage_names,
            output_storage_names=output_storage_names,
            input_step_names=input_step_names,
            output_step_names=output_step_names,
        )
        # Test without loader
        with pytest.raises(ValueError):
            step.process(**{k: 1.0 for k in input_step_names})
        # Test with loader
        db = {k: 1.0 for k in input_storage_names}
        loader = DefaultLoader(db)
        step.loader = loader
        cpd_logger.info(f"{input_storage_names},{output_storage_names},{input_step_names},{output_step_names}")
        result = step.process(**{k: 1.0 for k in input_step_names})
        assert isinstance(result, dict)

    @given(
        input_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=5),
            values=st.floats(allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=3,
        ),
        ref_type=st.sampled_from(["set", "dict"]),
    )
    def test_filter_and_rename(self, input_data, ref_type):
        from new_pysatl_cpd.steps.step import Step

        if not input_data:
            return
        if ref_type == "set":
            ref = set(input_data.keys())
            result = Step._filter_and_rename(input_data, ref)
            assert result == {k: input_data[k] for k in ref}
        else:
            ref = {k: f"renamed_{k}" for k in input_data}
            result = Step._filter_and_rename(input_data, ref)
            assert result == {f"renamed_{k}": v for k, v in input_data.items()}

    @pytest.mark.parametrize("has_loader", [True, False])
    def test_validate_storages(self, has_loader):
        builder = DummyReportBuilder(a=1.0, b=2.0)
        visualizer = DummyReportVisualizer()
        reporter = DummyReporter(report_builder=builder, report_visualizer=visualizer)
        step = ReportGenerationStep(reporter=reporter)
        if has_loader:
            step.loader = DefaultLoader({})
            assert step._validate_storages() is True
        else:
            step._loader = None
            assert step._validate_storages() is False

    def test_call_runs_process_and_validates(self):
        builder = MockReportBuilder()
        visualizer = MockReportVisualizer()
        reporter = MockReporter(report_builder=builder, report_visualizer=visualizer)
        step = ReportGenerationStep(reporter=reporter)
        step.loader = DefaultLoader({})
        # Should not raise
        result = step({})
        assert isinstance(result, dict)
        # Should raise if no loader
        step._loader = None
        with pytest.raises(ValueError):
            step({})

    def test_str(self):
        builder = DummyReportBuilder(a=1.0, b=2.0)
        visualizer = DummyReportVisualizer()
        reporter = DummyReporter(report_builder=builder, report_visualizer=visualizer)
        step = ReportGenerationStep(reporter=reporter, name="TestStep")
        assert str(step) == "TestStep (ReportGenerationStep)"
