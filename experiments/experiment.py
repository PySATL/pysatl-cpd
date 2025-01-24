from pathlib import Path
import logging
import datetime
from shutil import rmtree

from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from benchmarking.worker.optimal_threshold_worker import OptimalThresholdWorker
from benchmarking.worker.benchmarking_worker import BenchmarkingKNNWorker
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.report.benchmarking_report import BenchmarkingReport, Measures
from benchmarking.generator.generator import DistributionGenerator, Distribution


SAMPLE_COUNT_FOR_THRESHOLD_CALC = 70
WITHOUT_CP_SAMPLE_LENGTH = 200


class Experiment():
    def __init__(
        self,
        cpd_algorithm: BenchmarkingKNNAlgorithm,
        scrubber: BenchmarkingLinearScrubber,
        logger: logging.Logger
    ) -> None:
        """
        :param config_path: path to yaml file with distributions configuration.
        :param dataset_path: path to directory where generated datasets should be saved.
        :param dataset_path: path to directory where statistics should be saved.
        """
        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber
        self.__logger = logger

    def run_pipeline(self,
            significance_level: float,
            sample_count: int,
            interval_length: int,
            expected_change_points: list[int],
            config_path: Path,
            dataset_path: Path,
            results_path: Path,
            optimal_values_storage_path: Path,
            delta: float = 0.005
        ) -> None:

        distributions = DistributionGenerator.generate_by_config(config_path, dataset_path, sample_count)

        for i in range(len(distributions)):
            distr_comp = distributions[i]

            self.__logger.info(datetime.datetime.now())
            self.__logger.info("Distribution description start.")
            for distr in distr_comp:
                self.__logger.info(", ".join([f"type: {distr.type.name}", f"parameters: {distr.parameters}",  f"length: {distr.length}"]))
            self.__logger.info("Distribution description end.")

            # Generate the dataset without change points.
            without_cp_path = dataset_path / "without_cp"
            Path(without_cp_path).mkdir(parents=True, exist_ok=True)
            without_cp_distr = Distribution(distr_comp[0].type, distr_comp[0].parameters, WITHOUT_CP_SAMPLE_LENGTH)
            DistributionGenerator.generate([[without_cp_distr]], SAMPLE_COUNT_FOR_THRESHOLD_CALC, without_cp_path)

            # Calculate threshold on the dataset without change points and according to the given significance level.
            sample_length = sum(distr.length for distr in distr_comp)
            threshold_calculation = OptimalThresholdWorker(self.__cpd_algorithm, self.__scrubber, optimal_values_storage_path, significance_level, delta, sample_length, interval_length, self.__logger)
            threshold_calculation.run(without_cp_path, results_path)
            threshold = threshold_calculation.threshold

            assert threshold is not None, "Optimal threshold is None"

            # Removing the generated dataset without change points.
            rmtree(without_cp_path)

            # Run benchmark with calculated threshold.
            self.__cpd_algorithm.test_statistic = ThresholdOvercome(threshold)
            distr_path = dataset_path / Path(f"{i}-" + "-".join(map(lambda d: d.type.name, distr_comp)))
            cpd = BenchmarkingKNNWorker(self.__cpd_algorithm, self.__scrubber, expected_change_points)
            cpd.run(distr_path, results_path)

            report = BenchmarkingReport(results_path, expected_change_points, threshold, interval_length)
            report.add_scrubbing_alg_info()
            report.add_average_overall_time()
            report.add_average_window_time()
            report.add_power()
            evaluatedMetrics = report.get_result().filter_out_none()
            print(evaluatedMetrics)