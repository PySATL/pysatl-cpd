from new_pysatl_cpd.pipeline.pipeline import Pipeline
from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.steps.data_generation_step.data_handlers.generators.dummy_generator import DummyGenerator
from new_pysatl_cpd.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from new_pysatl_cpd.steps.experiment_execution_step.workers.dummy_worker import DummyWorker
from new_pysatl_cpd.steps.report_generation_step.report_builders.dummy_report_builder import DummyReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.dummy_report_visualizer import DummyReportVisualizer
from new_pysatl_cpd.steps.report_generation_step.reporters.dummy_reporter import DummyReporter

# save to gen.data. storage B={1:3}, add A={1:7} to metadata (step output)
step_1 = DataGenerationStep(DummyGenerator(), name="DummyGeneration")
# Get B as b, A as a from GenDataStorage. Save a+b as s to Result DB
step_2 = ExperimentExecutionStep(
    DummyWorker(), input_storage_names={"B": "b"}, input_step_names={"A": "a"}, name="DummyWorker"
)
# Generate Report with s from Result Storage
step_3 = ReportGenerationStep(
    DummyReporter(
        DummyReportBuilder(1, 2, builder_result_fields={"a", "b", "c", "s"}),
        DummyReportVisualizer(builder_result_fields={"a", "b", "c", "s"}),
    ),
    name="ReportGeneration",
)

steps = [step_1, step_2, step_3]
pipeline = Pipeline(steps)
pipeline.run()
