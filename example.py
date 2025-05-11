from new_pysatl_cpd.pipeline.pipeline import Pipeline
from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.steps.data_generation_step.data_handlers.generators.dummy_generator import DummyGenerator
from new_pysatl_cpd.steps.report_generation_step.report_builders.dummy_report_builder import DummyReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.dummy_report_visualizer import DummyReportVisualizer
from new_pysatl_cpd.steps.report_generation_step.reporters.reporter import Reporter
from new_pysatl_cpd.steps.test_execution_step.test_execution_step import TestExecutionStep
from new_pysatl_cpd.steps.test_execution_step.workers.dummy_worker import DummyWorker

step_1 = DataGenerationStep(DummyGenerator(), name="DummyGeneration")
step_2 = TestExecutionStep(DummyWorker(), name="DummyWorker")
step_3 = ReportGenerationStep(Reporter(DummyReportBuilder(1, 2), DummyReportVisualizer()), name="ReportGeneration")

steps = [step_1, step_2, step_3]
pipeline = Pipeline(steps)
pipeline.run()
