from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.problem import CpdProblem
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.online_cpd_solver import OnlineCpdSolver

algorithm_well_log = BayesianOnline(
    likelihood=HeuristicGaussianVsExponential(),
    hazard=ConstantHazard(4050),
    detector=ThresholdDetector(0.04),
    localizer=ArgmaxLocalizer(),
    learning_sample_size=20,
)

with open("well-log.txt") as f:
    well_log_data = [float(line.strip()) for line in f if line.strip()]

well_log_cpd_solver = OnlineCpdSolver(
    scenario=CpdProblem(True), algorithm=algorithm_well_log, algorithm_input=ListUnivariateProvider(well_log_data)
)

wellog_result = well_log_cpd_solver.run()
print(wellog_result)

algorithm_dji = BayesianOnline(
    likelihood=HeuristicGaussianVsExponential(),
    hazard=ConstantHazard(754),
    detector=ThresholdDetector(0.27),
    localizer=ArgmaxLocalizer(),
    learning_sample_size=20,
)

dji_data = pd.read_csv("DJIA.csv", parse_dates=["Date"])
dji_data["R_t"] = dji_data["Close"].pct_change()
returns_list = dji_data["R_t"].dropna().tolist()

dji_cpd_solver = OnlineCpdSolver(
    scenario=CpdProblem(True), algorithm=algorithm_dji, algorithm_input=ListUnivariateProvider(returns_list)
)

dji_result = dji_cpd_solver.run()
year_indices = dji_result.result
years = dji_data.iloc[year_indices]["Date"]

print(dji_result)
print(years)

algorithm_coal = BayesianOnline(
    likelihood=HeuristicGaussianVsExponential(),
    hazard=ConstantHazard(112),
    detector=ThresholdDetector(0.04),
    localizer=ArgmaxLocalizer(),
    learning_sample_size=20,
)


with open("coal-mining.csv") as f:
    coal_data = [float(line.strip()) for line in f if line.strip()]

coal_cpd_solver = OnlineCpdSolver(
    scenario=CpdProblem(True), algorithm=algorithm_coal, algorithm_input=ListUnivariateProvider(coal_data)
)

coal_result = coal_cpd_solver.run()
coal_years = [year + 1851 for year in coal_result.result]

print(coal_result)
print(coal_years)

output_dir = Path("detection_results")
output_dir.mkdir(exist_ok=True)

# Визуализация для Well-Log
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(well_log_data, label="Данные Well Log")

START, END = 1600, 2700

plt.xlim(START, END)

for cp in wellog_result.result:
    if START <= cp <= END:
        plt.axvline(cp, color="red", linestyle="--", linewidth=3.0, alpha=0.7)

legend_elements = [
    Line2D([0], [0], color="red", linestyle="--", label="Локализованная разладка"),
    Line2D([0], [0], color="blue", label="Данные Well Log"),
]

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.title("Разладки в данных Well Log (1600-2700)", fontsize=14)
plt.xlabel("Номер наблюдения", fontsize=12)
plt.ylabel("Значение", fontsize=12)
plt.legend(handles=legend_elements)  # Используем кастомную легенду
plt.grid(True)
plt.savefig(output_dir / "well_log_change_points.pdf", bbox_inches="tight")
plt.close()

# Визуализация для DJIA
# ---------------------------------------------------------------------
expected_dates = pd.to_datetime(["1973-01-01", "1973-10-19", "1974-08-09"])
plt.figure(figsize=(14, 6))

plt.plot(dji_data["Date"], dji_data["R_t"], color="blue", alpha=0.7, label="Дневная доходность")

ax = plt.gca()
y_min, y_max = ax.get_ylim()

ax.tick_params(axis="both", which="major", labelsize=11)
ax.tick_params(axis="both", which="minor", labelsize=11)

# Найденные точки изменений (красные линии)
for date in years:
    # Находим индекс ближайшей даты в данных
    idx = np.argmin(np.abs(dji_data["Date"] - date))
    plt.axvline(date, color="red", linestyle="--", alpha=0.7, linewidth=3, label="Локализованная разладка")
    plt.text(
        date,
        y_min + 0.06 * (y_max - y_min),
        f"{idx}",
        color="red",
        ha="left",
        va="top",
        fontsize=14,
        backgroundcolor=(1, 1, 1, 0.5),
    )

# Ожидаемые точки изменений (зеленые линии)
for date in expected_dates:
    # Находим индекс ближайшей даты в данных
    idx = np.argmin(np.abs(dji_data["Date"] - date))
    plt.axvline(date, color="green", linestyle=":", linewidth=3, alpha=0.9, label="Ожидаемая разладка")
    plt.text(
        date,
        y_min + 0.08 * (y_max - y_min),
        f"{idx}",
        color="green",
        ha="right",
        va="bottom",
        fontsize=14,
        backgroundcolor=(1, 1, 1, 0.5),
    )

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.title("Разладки в дневной доходности DJI", fontsize=14)
plt.xlabel("Дата", fontsize=12)
plt.ylabel("Дневная доходность", fontsize=12)

# Убираем дубликаты в легенде
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
filtered_labels = {
    "Дневная доходность": unique_labels["Дневная доходность"],
    "Локализованная разладка": unique_labels["Локализованная разладка"],
    "Ожидаемая разладка": unique_labels["Ожидаемая разладка"],
}

plt.legend(filtered_labels.values(), filtered_labels.keys(), prop={"size": 12})
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / "dji_returns_change_points.pdf", bbox_inches="tight")
plt.close()

# Визуализация для Coal Mining
# ---------------------------------------------------------------------
STARTING_YEAR = 1851
EXPECTED_CP = 1887  # 1887 - 1851 = 36

plt.figure(figsize=(14, 6))
plt.plot(list(range(STARTING_YEAR, STARTING_YEAR + len(coal_data))), coal_data, label="Катастрфы в угольных шахтах")

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

for cp in coal_result.result:
    plt.axvline(
        STARTING_YEAR + cp, color="red", linestyle="--", linewidth=3, alpha=0.7, label="Локализованная разладка"
    )

plt.axvline(EXPECTED_CP, color="green", linestyle=":", linewidth=3, alpha=0.7, label="Ожидаемая разладка (1887)")

plt.title("Разладки в данных Coal Mining Disasters", fontsize=14)
plt.xlabel("Год", fontsize=12)
plt.ylabel("Количество катастроф", fontsize=12)
plt.legend(prop={"size": 12})
plt.grid(True)
plt.savefig(output_dir / "coal_mining_change_points.pdf", bbox_inches="tight")
plt.close()

print("Результаты обнаружения точек изменений:")
print("Well Log - Обнаружено точек:", len(wellog_result.result))
print("DJI - Обнаружено точек:", len(dji_result.result))
print("Ожидаемые даты для DJI:", [d.strftime("%Y-%m-%d") for d in expected_dates])
print("\nCoal Mining - Обнаружено точек:", len(coal_result.result))
print("Ожидаемая точка для Coal Mining: 1887 год (36-й индекс)")
