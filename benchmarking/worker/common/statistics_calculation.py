"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import os
from pathlib import Path
from shutil import copy

import yaml

from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.algorithms.benchmarking_classification import BenchmarkingClassificationAlgorithm
from benchmarking.generator.generator import VerboseSafeDumper
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.worker.common.utils import Utils
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPContainer, CPDProblem


class StatisticsCalculation:
    @staticmethod
    def calculate_statistics(
        cpd_algorithm: BenchmarkingKNNAlgorithm | BenchmarkingClassificationAlgorithm,
        scrubber: BenchmarkingLinearScrubber,
        datasets_dir: Path,
        dest_dir: Path,
    ) -> list[CPContainer]:
        """
        :param datasets_dir: Path where datasets are stored. (".../n-distr-distr/")
        """
        # Generate algorithm and scrubber config in the root dir.
        alg_metaparams = cpd_algorithm.get_metaparameters()
        scrubber_metaparams = scrubber.get_metaparameters()
        config = {"algorithm": alg_metaparams, "scrubber": scrubber_metaparams}

        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / "config.yaml", "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)

        samples_dirs = Utils.get_all_sample_dirs(datasets_dir)
        stats_dirs = map(lambda x: x.name, Utils.get_all_stats_dirs(dest_dir))
        results = []

        for sample_dir in samples_dirs:
            if sample_dir[0].name in stats_dirs:
                print("done")
                continue

            data = LabeledCPData.read_generated_datasets(sample_dir[0])[sample_dir[1]].raw_data
            shell = CPDProblem(data, cpd_algorithm=cpd_algorithm, scrubber=scrubber)
            results.append(shell.run_cpd())

            bench_info = cpd_algorithm.get_benchmarking_info()

            dest_path = dest_dir / sample_dir[0].name
            dest_path.mkdir(parents=True, exist_ok=True)

            # Copy config of distribution one for all samples with the same distribution.
            if len(os.listdir(dest_dir)) == 1:
                # copy(datasets_dir / sample_dir[1] / "config.yaml", dest_path / sample_dir[1])
                copy(datasets_dir / "config.yaml", dest_path / sample_dir[1])

            with open(dest_path / "stats", "w") as outfile:
                for window_info in bench_info:
                    for stat in window_info.quality_statistics:
                        outfile.write(str(stat) + "\n")

            # Save time and memory. TODO: Save memory.
            overall_time = results[-1].time_sec
            avg_time = sum(map(lambda x: x.time, bench_info)) / len(bench_info)
            bench_info_format = {"overall_time": overall_time, "avg_time": avg_time}
            with open(dest_path / "benchmarking_info.yaml", "w") as outfile:
                yaml.dump(
                    bench_info_format, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper
                )

            # with open(dest_path / "benchmarking_info", "w") as outfile:
            #     for window_info in bench_info:
            #         for stat in window_info:
            #             outfile.write(str(stat) + "\n")

            print(sample_dir[0])

        return results

        # for sample_n in range(100):
        #     sample_dir = datasets_dir / f"sample_{sample_n}"

        #     data = LabeledCPData.read_generated_datasets(sample_dir)[datasets_dir.name].raw_data
        #     shell = CPDProblem(data, cpd_algorithm=cpd_algorithm, scrubber=scrubber)
        #     results.append(shell.run_cpd())

        #     bench_info = cpd_algorithm.get_benchmarking_info()

        #     dest_path = dest_dir / f"sample_{sample_n}"
        #     dest_path.mkdir(parents=True, exist_ok=True)

        #     # Copy config of distribution one for all samples with the same distribution.
        #     if len(os.listdir(dest_dir)) == 1:
        #         # copy(datasets_dir / sample_dir[1] / "config.yaml", dest_path / sample_dir[1])
        #         copy(datasets_dir / "config.yaml", dest_dir)

        #     with open(dest_path / "stats", "w") as outfile:
        #         for window_info in bench_info:
        #             for stat in window_info.quality_statistics:
        #                 outfile.write(str(stat) + "\n")

        #     # Save time and memory. TODO: Save memory.
        #     overall_time = results[-1].time_sec
        #     avg_time = sum(map(lambda x: x.time, bench_info)) / len(bench_info)
        #     bench_info_format = {"overall_time": overall_time, "avg_time": avg_time}
        #     with open(dest_path / "benchmarking_info.yaml", "w") as outfile:
        #         yaml.dump(
        #             bench_info_format, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper
        #         )

        #     # with open(dest_path / "benchmarking_info", "w") as outfile:
        #     #     for window_info in bench_info:
        #     #         for stat in window_info:
        #     #             outfile.write(str(stat) + "\n")

        #     print(sample_dir)

        # return results