from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from CPDShell.Core.algorithms.BayesianCPD.detectors.drop_detector import DropDetector
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer
from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell

THRESHOLD = 0.0
NUM_OF_SAMPLES = 1000
SAMPLE_SIZE = 500
BERNOULLI_PROB = 1.0 - 0.5 ** (1.0 / SAMPLE_SIZE)
HAZARD_RATE = 1 / BERNOULLI_PROB
LEARNING_SAMPLE_SIZE = 50

WORKING_DIR = Path()

CHANGE_POINT = 249
CHANGE_POINT_TOLERANCE = 25

def plot_algorithm_output(
        observations_data, change_points, run_lengths, gap_sizes, distributions, num, result_dir, with_save=False
):
    size = len(observations_data)
    plt.rcParams.update({'font.size': 18})

    fig, axes = plt.subplots(5, 1, figsize=(20, 10))

    ax1, ax2, ax3, ax4, ax5 = axes

    ax1.set_title(f"Data: {distributions} №{num}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.scatter(range(0, size), observations_data)
    ax1.plot(range(0, size), observations_data)
    ax1.set_xlim([0, size])
    ax1.margins(0)

    ax2.set_title("Run lengths distributions (log-normalized colors; white is 0.0 and black is 1.0)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("R.l. distribution")
    ax2.imshow(np.flip(np.rot90(run_lengths), 0), aspect="auto", cmap="gray_r", norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, size])
    ax2.yaxis.set_inverted(True)
    ax2.set_ylim([0, size])
    ax2.margins(0)

    ax3.set_title("Maximal run length's probability")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Probability")
    ax3.plot(
        run_lengths[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)],
        color="orange",
        label="Probability of the maximal run length (a.k.a. gap size)",
    )
    ax3.set_xlim([0, size])
    ax3.set_ylim([-0.1, 1.1])
    ax3.margins(0)
    ax3.grid()
    # ax3.legend()

    ax4.set_title("Is maximal run length's probability greater than 0 run length's probability?")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Probability")
    ax4.plot(
        (run_lengths[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)] > run_lengths[np.arange(SAMPLE_SIZE), 0]),
        color="red",
        label="Maximal run length's probability vs Zero run length's probability",
    )
    ax4.set_xlim([0, size])
    ax4.set_ylim([-0.1, 1.1])
    ax4.margins(0)
    ax4.grid()

    ax5.set_title("The most probable run length")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Run length")
    ax5.plot(np.argmax(run_lengths, axis=1), color="blue", label="The most probable run length")
    ax5.set_xlim([0, size])
    _, _, _, ymax = ax5.axis()
    ax5.set_ylim([0.0, 1.1 * ymax])
    ax5.margins(0)
    ax5.grid()

    for ax in axes:
        for cp in change_points:
            ax.axvline(cp, c="red", ls="dotted")

    plt.tight_layout()

    if with_save:
        if not Path(result_dir).exists():
            Path(result_dir).mkdir()
        plt.savefig(result_dir / f"{distributions}_{num}.png")
    else:
        plt.show()

    plt.close()


def process_normal_data():
    data = np.load("normal_data_for_validation.npy")
    max_rl_prob_matrix = np.zeros(data.shape)
    print(data.shape)

    for dataset_num in range(data.shape[0]):
        print(dataset_num, " / ", data.shape[0])

        gaussian_likelihood = GaussianUnknownMeanAndVariance()
        constant_hazard = ConstantHazard(HAZARD_RATE)
        simple_detector = SimpleDetector(THRESHOLD)
        simple_localizer = SimpleLocalizer()
        bayesian_algorithm = BayesianAlgorithm(
            learning_steps=LEARNING_SAMPLE_SIZE,
            likelihood=gaussian_likelihood,
            hazard=constant_hazard,
            detector=simple_detector,
            localizer=simple_localizer,
        )

        cpd = CPDShell(data=data[dataset_num], cpd_algorithm=bayesian_algorithm)
        cpd.scrubber.window_length = SAMPLE_SIZE
        cpd.scrubber.movement_k = 1.0
        bayesian_algorithm.localize(data[dataset_num])

        max_rl_prob = bayesian_algorithm.run_lengths[np.arange(SAMPLE_SIZE), bayesian_algorithm.gap_sizes.astype(int)]
        max_rl_prob[:LEARNING_SAMPLE_SIZE] = 1.0

        max_rl_prob_matrix[dataset_num] = max_rl_prob
        np.save("normal_max_rl_prob", max_rl_prob_matrix)


def generate_normal_data_for_validation():
    np.random.seed(42)

    means = np.random.rand(5) * 10
    variances = np.random.rand(5) * 5

    params = [(mean, var) for mean in means for var in variances]

    datasets = []
    for (mean, var) in params:
        data_samples = np.random.normal(loc=mean, scale=np.sqrt(var), size=(100, 500))
        datasets.append(data_samples)

    combined_data = np.vstack(datasets)

    np.save('normal_data_for_validation.npy', combined_data)

    for (mean, var), data_samples in zip(params, datasets):
        print(f'Mean: {mean:.2f}, Variance: {var:.2f}, Data shape: {data_samples.shape}')

    print(f'Combined data shape: {combined_data.shape}')

def process_samples(distributions, start, end, experiment_base_dir, results_base_dir):
    max_rl_prob_matrix = np.zeros((end - start, 500))
    for sample_num in range(start, end):
        print(distributions, sample_num)
        reader = LabeledCPData.read_generated_datasets(experiment_base_dir / f"{distributions}\\sample_{sample_num}")

        gaussian_likelihood = GaussianUnknownMeanAndVariance()
        constant_hazard = ConstantHazard(HAZARD_RATE)
        simple_detector = SimpleDetector(THRESHOLD)
        simple_localizer = SimpleLocalizer()
        bayesian_algorithm = BayesianAlgorithm(
            learning_steps=LEARNING_SAMPLE_SIZE,
            likelihood=gaussian_likelihood,
            hazard=constant_hazard,
            detector=simple_detector,
            localizer=simple_localizer,
        )

        data = reader[f"{distributions}"].raw_data
        cpd = CPDShell(reader[f"{distributions}"], cpd_algorithm=bayesian_algorithm)
        cpd.scrubber.window_length = SAMPLE_SIZE
        cpd.scrubber.movement_k = 1.0
        bayesian_algorithm.localize(data)

        result_dir = results_base_dir / f"{distributions}\\sample_{sample_num}"
        result_dir.mkdir(parents=True, exist_ok=True)

        max_rl_prob = bayesian_algorithm.run_lengths[np.arange(SAMPLE_SIZE), bayesian_algorithm.gap_sizes.astype(int)]
        max_rl_prob[:LEARNING_SAMPLE_SIZE] = 1.0

        max_rl_prob_matrix[sample_num] = max_rl_prob

        np.save(result_dir / "run_lengths", bayesian_algorithm.run_lengths)
        np.save(result_dir / "gap_sizes", bayesian_algorithm.gap_sizes)
        np.save(result_dir / "max_rl_prob", max_rl_prob)

        """
        plot_algorithm_output(
            data=data,
            change_points=reader[distributions].change_points,
            run_lengths=bayesian_algorithm.run_lengths,
            gap_sizes=bayesian_algorithm.gap_sizes,
            distributions=distributions,
            num=sample_num,
            result_dir=result_dir,
            with_save=True,
        )
        """

    return max_rl_prob_matrix


def process_datasets():
    # File paths to datasets and results directories.
    experiment_base_dir = Path("")
    results_base_dir = Path("")

    experiment_description = pd.read_csv(experiment_base_dir / "experiment_description.csv")
    names = experiment_description["name"].tolist()

    max_rl_prob_matrices = []
    for name in set(names).difference({"normal", "uniform", "exponential", "weibull", "beta"}):
        max_rl_probs_matrix = process_samples(name, 0, 1000, experiment_base_dir, results_base_dir)
        max_rl_prob_matrices.append(max_rl_probs_matrix)

    overall_matrix = np.concatenate(max_rl_prob_matrices, axis=0)
    np.save(results_base_dir / "max_rl_probs_with_cp", overall_matrix)

def calculate_powers_for_data_configuration(dataframes, configuration_name, thresholds, start, end):
    detection_df, localization_df = dataframes

    dir_path = Path("") / configuration_name
    for threshold in thresholds:
        detection_hypothesis_positives = 0
        localization_hypothesis_positives = 0
        for data_num in range(start, end):
            data_path = dir_path / f"sample_{data_num}"
            rl_distributions = np.load(data_path / "run_lengths.npy")
            gap_sizes = np.load(data_path / "gap_sizes.npy")
            cps = analyze_cp(rl_distributions, gap_sizes, threshold)
            if len(cps) > 0:
                detection_hypothesis_positives += 1
                # print(cps[0])
                if cps[0] in range(CHANGE_POINT - CHANGE_POINT_TOLERANCE, CHANGE_POINT + CHANGE_POINT_TOLERANCE + 1):
                    localization_hypothesis_positives += 1
                else:
                    print(threshold, f"CHANGE POINT: {cps[0]} |", configuration_name, data_num)

        detection_df.loc[configuration_name, threshold] = detection_hypothesis_positives / (end - start)
        localization_df.loc[configuration_name, threshold] = localization_hypothesis_positives / (end - start)

def calculate_powers(thresholds, start, end):
    experiment_base_dir = Path("")

    experiment_description = pd.read_csv(experiment_base_dir / "experiment_description.csv")
    names = experiment_description["name"].tolist()

    detection_df = pd.DataFrame(columns=thresholds)
    localization_df = pd.DataFrame(columns=thresholds)

    for name in {"normal", "uniform", "exponential", "weibull", "beta"}:
        initial_powers = [0.0] * len(thresholds)
        detection_df.loc[name] = initial_powers
        localization_df.loc[name] = initial_powers
        dfs = (detection_df, localization_df)
        calculate_powers_for_data_configuration(dfs, f"normal-{name}", thresholds, start, end)

    # detection_df.to_csv("detection_powers.csv")
    # localization_df.to_csv("localization_powers.csv")

def analyze_cp(rl_distributions, gap_sizes, cp_threshold):
    max_rl_prob = rl_distributions[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)]
    max_rl_prob[:LEARNING_SAMPLE_SIZE] = 1.0

    cp_detection_times = np.where(max_rl_prob < cp_threshold)[0]
    cps = []
    if cp_detection_times.size > 0:
        first_cp_detection_time = cp_detection_times[0]
        gap_size = int(gap_sizes[first_cp_detection_time])
        run_length_candidates = rl_distributions[first_cp_detection_time][:gap_size]

        correct_run_length = run_length_candidates.argmax()
        cp = int(first_cp_detection_time - correct_run_length)
        cps.append(cp)

        if cps[0] not in range(CHANGE_POINT - CHANGE_POINT_TOLERANCE, CHANGE_POINT + CHANGE_POINT_TOLERANCE + 1):
            print("FIRST CP DETECTION TIME: ", first_cp_detection_time)

    return cps

def calculate_significance_levels():
    result_base_dir = Path("")
    for name in {"normal", "uniform", "exponential", "weibull", "beta"}:
        max_rl_probs_matrix = np.zeros((1000, 500))
        for sample_num in range (0, 1000):
            data_path = result_base_dir / name / f"sample_{sample_num}"
            rl_distributions = np.load(data_path / "run_lengths.npy")
            gap_sizes = np.load(data_path / "gap_sizes.npy")
            max_rl_prob = rl_distributions[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)]
            max_rl_prob[:LEARNING_SAMPLE_SIZE] = 1.0

            max_rl_probs_matrix[sample_num] = max_rl_prob

        np.save(result_base_dir / "max_rl_prob.npy", max_rl_probs_matrix)

        thresholds_for_significance_level = np.linspace(0.0, 1.0, 101)
        significance_levels = []

        for threshold in thresholds_for_significance_level:
            exceed_count = np.sum(np.any(max_rl_probs_matrix < threshold, axis=1))

            false_cp_probability = exceed_count / max_rl_probs_matrix.shape[0]
            significance_levels.append(false_cp_probability)

        df = pd.DataFrame({
            'Threshold': thresholds_for_significance_level,
            'Significance Level': np.array(significance_levels)
        })

        df.to_csv(result_base_dir / f"{name}\\significant_thresholds.csv")

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_for_significance_level, significance_levels, marker='o', linestyle='-', color='b')
        plt.title(f'{name}: significance levels for given threshold values')
        plt.xlabel('Thresholds')
        plt.ylabel('Significance levels')
        plt.grid(True)
        # plt.xticks(ticks=thresholds_for_significance_level)  # Установка меток по оси x
        # plt.yticks(ticks=significance_levels)  # Установка меток по оси y
        plt.xlim(0, 1)  # Ограничение оси x от 0 до 1
        plt.ylim(0, max(significance_levels) * 1.1)  # Ограничение оси y с учетом максимального значения

        # Сохранение графика в файл
        plt.savefig(result_base_dir / f"{name} \\ significance_levels_vs_thresholds.png")  # Сохранение в формате PNG
        # plt.savefig('significance_levels_vs_thresholds.pdf')  # Можно сохранить также в PDF, если это необходимо

        # Закрыть текущую фигуру (если не нужно показывать график)
        plt.close()

def build_bad_plots():
    save_path = Path("")
    cases = [
        ("normal-exponential", 745),
        ("normal-exponential", 83),
        ("normal-weibull", 364),
        ("normal-weibull", 383),
        ("normal-uniform", 823),
        ("normal-uniform", 831),
        ("normal-beta", 755),
        ("normal-beta", 748),
        ("normal-normal", 308),
        ("normal-normal", 65)
    ]

    for (name, num) in cases:
        data_path = Path(f"")
        reader = LabeledCPData.read_generated_datasets(data_path)
        data = reader[name].raw_data

        results_path = Path(f"")
        run_lengths = np.load(results_path / "run_lengths.npy")
        gap_sizes = np.load(results_path / "gap_sizes.npy")

        plot_algorithm_output(
            observations_data=data,
            change_points=reader[name].change_points,
            run_lengths=run_lengths,
            gap_sizes=gap_sizes,
            distributions=name,
            num=num,
            result_dir=save_path,
            with_save=True,
        )


if __name__ == "__main__":
    # generate_normal_data_for_validation()
    # process_normal_data()

    # process_datasets()

    # path = Path("")
    # data = pd.read_csv(path / "sample.csv").to_numpy()
    # print(data[248:251])

    # example()

    # thresholds_for_power = [0.27, 0.18, 0.07, 0.04]
    # calculate_powers(thresholds_for_power, 0, 1000)

    # calculate_significance_levels()

    build_bad_plots()

    """
    normal_data = np.load("normal_data_for_validation.npy")
    # max_rl_probs_matrix = np.load("")
    max_rl_probs_matrix = np.load("normal_max_rl_prob.npy")
    cp_data_example = np.concatenate((np.random.normal(0, 1, 250), np.random.normal(1, np.sqrt(2), 250)))
    test_random_data = scipy.stats.norm.rvs(loc=1.5601864, scale=np.sqrt(0.5), size=500)

    NORMAL_SAMPLE_NUM = 2173
    data = cp_data_example

    gaussian_likelihood = GaussianUnknownMeanAndVariance()
    constant_hazard = ConstantHazard(HAZARD_RATE)
    simple_detector = SimpleDetector(THRESHOLD)
    simple_localizer = SimpleLocalizer()
    drop_detector = DropDetector(0.9)

    bayesian_algorithm = BayesianAlgorithm(
        learning_steps=LEARNING_SAMPLE_SIZE,
        likelihood=gaussian_likelihood,
        hazard=constant_hazard,
        detector=simple_detector,
        localizer=simple_localizer,
    )

    cpd = CPDShell(data, cpd_algorithm=bayesian_algorithm)
    cpd.scrubber.window_length = SAMPLE_SIZE
    cpd.scrubber.movement_k = 1.0
    bayesian_algorithm.localize(data)

    plot_algorithm_output(
        observations_data=data,
        change_points=[],
        run_lengths=bayesian_algorithm.run_lengths,
        gap_sizes=bayesian_algorithm.gap_sizes,
        distributions="normal validation",
        num=NORMAL_SAMPLE_NUM,
        result_dir=Path(),
        with_save=False,
    )

    thresholds = np.linspace(0.0, 1.0, num=101)
    significance_levels = []

    for threshold in thresholds:
        exceed_count = np.sum(np.any(max_rl_probs_matrix < threshold, axis=1))

        false_cp_probability = exceed_count / max_rl_probs_matrix.shape[0]
        significance_levels.append(false_cp_probability)

    for threshold, probability in zip(thresholds, significance_levels):
        print(f'Threshold: {threshold}, Significance level: {probability}')"""
