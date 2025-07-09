from pathlib import Path

from benchmarking.pipeline.pipeline import Pipeline
from benchmarking.steps.data_generation_step.data_generation_step import DataGenerationStep
from benchmarking.steps.data_generation_step.data_handlers.generators.cpd_generator import CpdGenerator
from benchmarking.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from benchmarking.steps.experiment_execution_step.workers.run_complete_algorithm_worker import (
    RunCompleteAlgorithmWorker,
)
from benchmarking.steps.report_generation_step.report_builders.change_point_builder import CpBuilder
from benchmarking.steps.report_generation_step.report_generation_step import ReportGenerationStep
from benchmarking.steps.report_generation_step.report_visualizers.change_point_text_visualizer import CpTextVisualizer
from benchmarking.steps.report_generation_step.reporters.reporter import Reporter
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_algorithm import BayesianAlgorithm

# Generate data with example config and save as my_experiment_dataset
generator = CpdGenerator(
    name="cpd_generator", output_storage_names={"example"}, config=Path("examples/configs/test_config_exp.yml")
)
step_1 = DataGenerationStep(
    data_handler=generator,
    name="cpd_generation_test_config_exp_step",
    output_storage_names={"example": "my_experiment_dataset"},
)

# Initialize BayesianAlgorithm and run with generated data
algorithm = BayesianAlgorithm(
    learning_steps=5,
    likelihood=HeuristicGaussianVsExponential(),
    hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
    detector=ThresholdDetector(threshold=0.005),
    localizer=ArgmaxLocalizer(),
)
algo_worker = RunCompleteAlgorithmWorker(algorithm=algorithm, name="run_bayesian_algorithm_worker")
step_2 = ExperimentExecutionStep(
    worker=algo_worker, name="run_bayesian_algorithm_step", input_storage_names={"my_experiment_dataset": "dataset"}
)

# Generate text report with change points from Result Storage
builder = CpBuilder()
visualizer = CpTextVisualizer(file_name="my_experiment_change_points_report")
reporter = Reporter(builder, visualizer, name="text_reporter")
step_3 = ReportGenerationStep(reporter, name="ReportGeneration", input_storage_names={"change_points"})

steps = [step_1, step_2, step_3]
pipeline = Pipeline(steps)
pipeline.run()
