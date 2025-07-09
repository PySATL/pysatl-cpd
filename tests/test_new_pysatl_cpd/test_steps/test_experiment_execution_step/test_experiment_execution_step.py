import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarking.logger import cpd_logger
from benchmarking.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from benchmarking.steps.experiment_execution_step.workers.dummy_worker import DummyWorker
from benchmarking.storages.loaders.default_loader import DefaultLoader
from benchmarking.storages.savers.default_saver import DefaultSaver
from tests.test_new_pysatl_cpd.test_steps.test_experiment_execution_step.test_workers.mock_worker import MockWorker


class TestExperimentExecutionStep:
    @given(
        input_storage_names=st.sets(st.text()),
        output_storage_names=st.sets(st.text()),
        input_step_names=st.sets(st.text()),
        output_step_names=st.sets(st.text()),
    )
    def test_process_and_storage(self, input_storage_names, output_storage_names, input_step_names, output_step_names):
        # Setup
        worker = MockWorker(result={k: 1.00 for k in output_storage_names.union(output_step_names)})
        if input_step_names.intersection(input_storage_names) or output_step_names.intersection(output_storage_names):
            with pytest.raises(ValueError):
                step = ExperimentExecutionStep(
                    worker=worker,
                    input_storage_names=input_storage_names,
                    output_storage_names=output_storage_names,
                    input_step_names=input_step_names,
                    output_step_names=output_step_names,
                )
            return

        step = ExperimentExecutionStep(
            worker=worker,
            input_storage_names=input_storage_names,
            output_storage_names=output_storage_names,
            input_step_names=input_step_names,
            output_step_names=output_step_names,
        )

        # Test without loader/saver
        with pytest.raises(ValueError):
            step.process(**{k: 1.0 for k in input_step_names})
        # Test with loader and saver
        db = {k: 1.0 for k in input_storage_names}
        loader = DefaultLoader(db)
        saver = DefaultSaver({})
        step.loader = loader
        step.saver = saver
        cpd_logger.debug(f"{input_storage_names}, {output_storage_names}, {input_step_names}, {output_step_names}")
        result = step.process(**{k: 1.0 for k in input_step_names})
        assert isinstance(result, dict)
        for k in output_storage_names:
            assert k in saver.dict_as_db

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
        # Test the static method _filter_and_rename from Step
        from benchmarking.steps.step import Step

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

    @pytest.mark.parametrize("has_loader,has_saver", [(True, True), (True, False), (False, True), (False, False)])
    def test_validate_storages(self, has_loader, has_saver):
        worker = DummyWorker()
        step = ExperimentExecutionStep(worker=worker)
        if has_loader:
            step.loader = DefaultLoader({})
        if has_saver:
            step.saver = DefaultSaver({})
        assert step._validate_storages() is (has_loader and has_saver)

    def test_call_runs_process_and_validates(self):
        worker = MockWorker()
        step = ExperimentExecutionStep(worker=worker)
        step.loader = DefaultLoader({})
        step.saver = DefaultSaver({})
        # Should not raise
        result = step({})
        assert isinstance(result, dict)
        # Should raise if no loader
        step._loader = None
        with pytest.raises(ValueError):
            step({})

    def test_str(self):
        worker = DummyWorker()
        step = ExperimentExecutionStep(worker=worker, name="TestStep")
        assert str(step) == "TestStep (ExperimentExecutionStep)"
