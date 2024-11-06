from pathlib import Path

import numpy as np
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
from CPDShell.shell import CPDShell

THRESHOLD = 0.0
NUM_OF_SAMPLES = 1000
SAMPLE_SIZE = 500
BERNOULLI_PROB = 1.0 - 0.5 ** (1.0 / SAMPLE_SIZE)
HAZARD_RATE = 1 / BERNOULLI_PROB
LEARNING_SAMPLE_SIZE = 50

WORKING_DIR = Path()


def plot_algorithm_output(
        size, data, change_points, bayesian_algorithm, distributions, num, result_dir, with_save=False
):
    plt.rcParams.update({'font.size': 18})

    run_lengths = bayesian_algorithm.run_lengths
    gap_sizes = bayesian_algorithm.gap_sizes

    fig, axes = plt.subplots(5, 1, figsize=(20, 10))

    ax1, ax2, ax3, ax4, ax5 = axes

    ax1.set_title(f"Data: {distributions} №{num}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.scatter(range(0, size), data)
    ax1.plot(range(0, size), data)
    ax1.set_xlim([0, size])
    ax1.margins(0)

    ax2.set_title("Run lengths distributions (log-normalized colors; white is 0.0 and black is 1.0)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Run lengths distribution")
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
    ax3.margins(0)
    ax3.grid()
    # ax3.legend()

    ax4.set_title("Is maximal run length's probability greater than 0 run lengths' probability?")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Probability")
    ax4.plot(
        (run_lengths[np.arange(SAMPLE_SIZE), gap_sizes.astype(int)] > run_lengths[np.arange(SAMPLE_SIZE), 0]),
        color="red",
        label="Maximal run length's probability vs Zero run length's probability",
    )
    ax4.set_xlim([0, size])
    ax4.margins(0)
    ax4.grid()

    ax5.set_title("The most probable run length")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Run length")
    ax5.plot(np.argmax(run_lengths, axis=1), color="blue", label="The most probable run length")
    ax5.set_xlim([0, size])
    ax5.margins(0)

    for ax in axes:
        for cp in change_points:
            ax.axvline(cp, c="red", ls="dotted")

    plt.tight_layout()

    if with_save:
        if not Path(result_dir).exists():
            Path(result_dir).mkdir()
        plt.savefig(result_dir / (distributions + ".png"))
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


if __name__ == "__main__":
    generate_normal_data_for_validation()
    process_normal_data()

    normal_data = np.load("normal_data_for_validation.npy")
    max_rl_probs_matrix = np.load("normal_max_rl_prob.npy")
    cp_data_example = np.concatenate((np.random.normal(0, 1, 250), np.random.normal(1, 2, 250)))
    test_random_data = scipy.stats.norm.rvs(loc=1.5601864, scale=np.sqrt(1.73365944), size=500)

    NORMAL_SAMPLE_NUM = 2173

    data = normal_data[NORMAL_SAMPLE_NUM]

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
        size=SAMPLE_SIZE,
        data=data,
        change_points=[],
        bayesian_algorithm=bayesian_algorithm,
        distributions="normal validation",
        num=NORMAL_SAMPLE_NUM,
        result_dir=Path(),
        with_save=False,
    )

    thresholds = np.linspace(0.0, 0.1, num=10001)
    significance_levels = []

    for threshold in thresholds:
        exceed_count = np.sum(np.any(max_rl_probs_matrix < threshold, axis=1))

        false_cp_probability = exceed_count / max_rl_probs_matrix.shape[0]
        significance_levels.append(false_cp_probability)

    for threshold, probability in zip(thresholds, significance_levels):
        print(f'Threshold: {threshold}, Significance level: {probability}')
