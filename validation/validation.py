import pandas as pd

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

coal_data = pd.read_csv("DJIA.csv", parse_dates=["Date"])
coal_data["R_t"] = coal_data["Close"].pct_change()
returns_list = coal_data["R_t"].dropna().tolist()

dji_cpd_solver = OnlineCpdSolver(
    scenario=CpdProblem(True), algorithm=algorithm_dji, algorithm_input=ListUnivariateProvider(returns_list)
)

dji_result = dji_cpd_solver.run()
year_indices = dji_result.result
years = coal_data.iloc[year_indices]["Date"]

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
