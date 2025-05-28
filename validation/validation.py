from pathlib import Path

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


with open("coal-mining.pdf") as f:
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

# Well-Log
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(well_log_data, label="Well Log Data")

start, end = 1600, 2700
plt.xlim(start, end)

for cp in wellog_result.result:
    if start <= cp <= end:
        plt.axvline(cp, color="red", linestyle="--", alpha=0.7)

legend_elements = [
    Line2D([0], [0], color="red", linestyle="--", label="Detected Change Points"),
    Line2D([0], [0], color="blue", label="Well Log Data"),
]

plt.title(f"Well Log Change Points Detection ({start}--{end})")
plt.xlabel("Measurement Index")
plt.ylabel("Value")
plt.legend(handles=legend_elements)
plt.grid(True)
plt.savefig(output_dir / "well_log_change_points.pdf", bbox_inches="tight")
plt.close()

# DJI
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(dji_data["Date"], dji_data["R_t"], color="blue", alpha=0.7, label="Daily Returns")

for date in years:
    plt.axvline(date, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="Detected Change Point")

plt.title("DJI Returns")
plt.xlabel("Measurement Index")
plt.ylabel("Value")
plt.legend(handles=legend_elements)
plt.grid(True)
plt.savefig(output_dir / "dji_returns.pdf", bbox_inches="tight")
plt.close()
