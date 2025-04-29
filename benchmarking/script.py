from pathlib import Path

import pandas as pd

from benchmarking.worker import Worker
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider


def construct_algorithm(threshold, learning_sample_size):
    likelihood = HeuristicGaussianVsExponential()
    return BayesianOnline(
        learning_sample_size=learning_sample_size,
        likelihood=likelihood,
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=threshold),
        localizer=ArgmaxLocalizer(),
    ), likelihood


def benchmark(configurations, num_of_experiments, parameters, input_path, output_path):
    columns = [
        "configuration",
        "experiment_number",
        "threshold",
        "learning_sample_size",
        "change_point",
        "detected_change_points",
        "detection_delays",
        "work_time",
        "likelihoods_history",
    ]

    save_path = Path(output_path) / "experiment_results.csv"
    history_df = pd.read_csv(save_path, keep_default_na=False) if save_path.exists() else pd.DataFrame(columns=columns)

    combination_count = len(configurations) * len(parameters)
    progress_counter = 0

    for threshold, learning_sample_size in parameters:
        for configuration in configurations:
            print("%.2f" % (progress_counter * 100 / combination_count), "%")
            progress_counter += 1

            buffer = []
            for experiment in range(num_of_experiments):
                if not history_df[
                    (history_df["configuration"] == configuration)
                    & (history_df["threshold"] == threshold)
                    & (history_df["learning_sample_size"] == learning_sample_size)
                    & (history_df["experiment_number"] == experiment)
                ].empty:
                    continue

                data_path = Path(f"{input_path}/experiment/stage_1/{configuration}/sample_{experiment}/{configuration}")
                change_points = 250 if "-" in configuration else None

                data = pd.read_csv(data_path / "sample.csv").to_numpy()
                data_provider = ListUnivariateProvider(data=list(data))

                algorthm, likelihood = construct_algorithm(
                    threshold=threshold, learning_sample_size=learning_sample_size
                )
                cpd_core = OnlineCpdCore(algorithm=algorthm, data_provider=data_provider)
                worker = Worker(online_cpd=cpd_core)

                cp_results, cpd_time = worker.run()

                detected_change_points = [result.change_point for result in cp_results]
                detection_delays = [result.delay for result in cp_results]

                buffer.append(
                    {
                        "configuration": configuration,
                        "experiment_number": experiment,
                        "threshold": threshold,
                        "learning_sample_size": learning_sample_size,
                        "change_point": change_points,
                        "detected_change_points": detected_change_points,
                        "detection_delays": detection_delays,
                        "work_time": cpd_time,
                        "likelihoods_history": likelihood.likelihoods_history,
                    }
                )

            if len(buffer) > 0:
                pd.DataFrame(buffer).to_csv(save_path, mode="a", header=not save_path.exists(), index=False)


def run_benchmark():
    # Data: https://drive.google.com/drive/folders/1cp0V29ZTuOW0tqDAdSiaDHopFhOtfVPj?usp=sharing
    input_path = "input/path"
    output_path = "output/path"

    thresholds = [0.27, 0.18, 0.07, 0.04]
    learning_sample_sizes = [50, 20, 10, 5]
    parameters_grid = [(threshold, sample_size) for threshold in thresholds for sample_size in learning_sample_sizes]

    configurations = list(
        pd.read_csv(Path(input_path) / "experiment\\stage_1\\experiment_description")["name"].to_list()
    )
    num_of_experiments = 1000

    benchmark(
        configurations=configurations,
        num_of_experiments=num_of_experiments,
        parameters=parameters_grid,
        input_path=input_path,
        output_path=output_path,
    )


run_benchmark()
