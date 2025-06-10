import pickle
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarking.worker import Worker
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import GaussianConjugate
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider


def construct_algorithm_for_hists(likelihood):
    return BayesianOnline(
        learning_sample_size=20,
        likelihood=likelihood,
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=0.18),
        localizer=ArgmaxLocalizer(),
    )


def compute_delays(input_path, results_file="delays_results.pkl"):
    """Вычисляет задержки и сохраняет результаты в файл"""
    configurations = ["normal-normal"]
    all_results = {}

    for configuration in configurations:
        results = {}
        models = [
            (GaussianConjugate(), "Гауссовская модель"),
            # (ExponentialConjugate(), "Экспоненциальная модель"),
            (HeuristicGaussianVsExponential(), "Эвристическая модель"),
        ]

        for model, model_name in models:
            results[model_name] = []
            for experiment in range(1000):
                try:
                    data_path = Path(
                        rf"{input_path}\experiment\stage_1\{configuration}\sample_{experiment}\{configuration}"
                    )
                    if not data_path.exists():
                        continue

                    data = pd.read_csv(data_path / "sample.csv").to_numpy()
                    algorithm = construct_algorithm_for_hists(model)
                    data_provider = ListUnivariateProvider(data=list(data))

                    cpd_core = OnlineCpdCore(algorithm=algorithm, data_provider=data_provider)
                    worker = Worker(online_cpd=cpd_core)

                    cp_results, _ = worker.run()
                    if cp_results:
                        results[model_name].extend([result.delay for result in cp_results])

                except Exception as e:
                    print(f"Error processing {configuration}/{experiment}: {e!s}")
                    continue

        all_results[configuration] = results
        print(f"Completed processing for configuration: {configuration}")

    # Сохраняем результаты в файл
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {results_file}")

    return all_results


def load_delays(results_file="delays_results_norm.pkl"):
    """Загружает результаты из файла"""
    with open(results_file, "rb") as f:
        return pickle.load(f)


def plot_hists_from_results(results, output_dir):
    """Строит гистограммы для презентаций с оптимизированным форматированием"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Параметры для презентации (половина слайда)
    mpl.rcParams.update(mpl.rcParamsDefault)

    plt.rcParams.update(
        {
            "axes.titlesize": 24,  # Уменьшенный размер
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 26,
            #'font.family': 'sans-serif',
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )

    configuration_notations = {
        "exponential-exponential": r"$\mathrm{Exp}–\mathrm{Exp}$",
        "weibull-weibull": r"$W–W$",
        "normal-normal": r"$\mathcal{N}–\mathcal{N}$",
    }

    for configuration, config_results in results.items():
        models = list(config_results.keys())

        # Компактный размер для половины слайда
        fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(10, 10))
        fig.suptitle(f"Распределение задержек: {configuration_notations[configuration]}", fontsize=26, y=0.95)

        # Фиксированные параметры для лучшего масштабирования
        X_LIM = (0, 150)  # Обрезаем длинные задержки
        BIN_WIDTH = 5  # Ширина бина

        for ax, (model_name, delays) in zip(axes, config_results.items()):
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_title(model_name, fontsize=22)
            ax.set_xlabel("Задержка", labelpad=8)
            ax.set_ylabel("Частота", labelpad=8)  # Более компактное название

            # Установка фиксированных границ
            ax.set_xlim(X_LIM)
            bin_edges = np.arange(X_LIM[0], X_LIM[1] + BIN_WIDTH, BIN_WIDTH)

            if not delays:
                ax.text(0.5, 0.5, "Нет данных", fontsize=20, ha="center", va="center")
                ax.set_ylim(0, 10)
                continue

            # Фильтрация данных в пределах X_LIM
            filtered_delays = [d for d in delays if X_LIM[0] <= d <= X_LIM[1]]

            # Гистограмма
            counts, bins, patches = ax.hist(
                filtered_delays,
                bins=bin_edges,
                edgecolor="black",
                alpha=0.7,
                color="#1f77b4",  # Единый цвет для всех
            )

            # Квантили с оптимизированным размещением
            quantiles = np.quantile(delays, [0.5, 0.75, 0.9])
            colors = ["#2ca02c", "#ff7f0e", "#d62728"]
            labels = ["50%", "75%", "90%"]

            # Уникальные высоты для каждой метки (в порядке квантилей)
            height_levels = [0.85, 0.15, 0.65]  # 50%:верх, 75%:низ, 90%:середина
            va_list = ["top", "bottom", "top"]  # Вертикальное выравнивание

            max_count = max(counts) if counts.any() else 1

            for idx, (q, color, label) in enumerate(zip(quantiles, colors, labels)):
                if q < X_LIM[1]:
                    ax.axvline(q, color=color, linestyle="--", linewidth=3)

                    # Выбираем высоту из предопределенных уровней
                    y_pos = max_count * height_levels[idx]
                    va = va_list[idx]

                    # Для 90% смещаем текст влево если он близко к правому краю
                    horizontal_offset = 1
                    ha = "left"
                    if label == "90%" and q > X_LIM[1] * 0.8:
                        horizontal_offset = -1
                        ha = "right"

                    ax.text(
                        q + horizontal_offset,
                        y_pos,
                        f"{label}: {q:.1f}",
                        color=color,
                        fontsize=18,
                        ha=ha,
                        va=va,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                    )

            # Автомасштабирование по Y с ограничением
            y_max = max_count * 1.25
            ax.set_ylim(0, y_max if y_max > 10 else 10)

        # Компактное расположение
        plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
        plt.savefig(output_dir / f"{configuration}_hists_presentation.pdf", bbox_inches="tight", transparent=False)
        plt.close()

    print(f"Presentation histograms saved to {output_dir}")
    # Пример использования:


input_path = ""
output_dir = ""
results_file = "delays_results.pkl"

# Вариант 1: Вычислить и сохранить результаты + построить графики
# results = compute_delays(input_path, results_file)
# plot_hists_from_results(results, output_dir)

# Вариант 2: Загрузить ранее сохраненные результаты и построить графики
results = load_delays(results_file)
plot_hists_from_results(results, output_dir)
