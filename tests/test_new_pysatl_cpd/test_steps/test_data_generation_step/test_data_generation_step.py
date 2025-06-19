import pytest
from hypothesis import given
from hypothesis import strategies as st

from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.storages.savers.default_saver import DefaultSaver
from tests.test_new_pysatl_cpd.test_steps.test_data_generation_step.test_data_handlers.mock_data_handler import (
    MockDataHandler,
)


class TestDataGenerationStep:
    @pytest.mark.parametrize(
        "input_step_names,output_storage_names,num_chunks,values_per_chunk",
        [
            (set(), set(), 1, 1),
            ({"meta1"}, {"value_0"}, 2, 1),
            ({"meta1", "meta2"}, {"value_0", "value_1"}, 3, 2),
        ],
    )
    def test_process_and_storage(self, input_step_names, output_storage_names, num_chunks, values_per_chunk):
        # Setup
        handler = MockDataHandler(
            input_step_names=input_step_names,
            output_storage_names=output_storage_names,
            num_chunks=num_chunks,
            values_per_chunk=values_per_chunk,
        )
        step = DataGenerationStep(
            data_handler=handler, input_step_names=input_step_names, output_storage_names=output_storage_names
        )
        # Test without saver
        result = step.process(**{k: 1.0 for k in input_step_names})
        assert isinstance(result, dict)
        # Test with saver
        db = {}
        saver = DefaultSaver(db)
        step.saver = saver
        result = step(**{k: 1.0 for k in input_step_names})
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

    @pytest.mark.parametrize("has_saver", [True, False])
    def test_validate_storages(self, has_saver):
        handler = MockDataHandler()
        step = DataGenerationStep(data_handler=handler)
        if has_saver:
            step.saver = DefaultSaver({})
            assert step._validate_storages() is True
        else:
            step._saver = None
            assert step._validate_storages() is False

    def test_call_runs_process_and_validates(self):
        handler = MockDataHandler()
        step = DataGenerationStep(data_handler=handler)
        step.saver = DefaultSaver({})
        # Should not raise
        result = step({})
        assert isinstance(result, dict)
        # Should raise if no saver
        step._saver = None
        with pytest.raises(ValueError):
            step({})

    def test_str(self):
        handler = MockDataHandler()
        step = DataGenerationStep(data_handler=handler, name="TestStep")
        assert str(step) == "TestStep (DataGenerationStep)"
