# PySATL-CPD

[status-shield]: https://img.shields.io/github/actions/workflow/status/PySATL/pysatl-cpd/.github/workflows/check.yaml?branch=main&event=push&style=for-the-badge&label=Checks
[status-url]: https://github.com/PySATL/pysatl-cpd/blob/main/.github/workflows/check.yaml
[license-shield]: https://img.shields.io/github/license/PySATL/pysatl-cpd.svg?style=for-the-badge&color=blue
[license-url]: LICENSE

[![Checks][status-shield]][status-url]
[![MIT License][license-shield]][license-url]

**Change point detection** module (*abbreviated CPD module*) is a module, designed for detecting anomalies in time series data, which refer to significant deviations from expected patterns or trends. Anomalies can indicate unusual events or changes in a system, making them crucial for monitoring and analysis in various fields such as finance, healthcare, and network security.

At the moment, the module implements the following CPD algorithms:
* Bayesian algorithm (scrubbing and online versions)
* Density based algorithms:
    * KLIEP
    * RuLSIF
* Graph algorithm
* k-NN based algorithm
* Algorithms, based on classifiers:
    * SVM
    * KNN
    * Decision Tree
    * Logistic Regression
    * Random Forest
---

## Requirements

- Python 3.10+
- Poetry 1.8.0+

## Installation

Clone the repository:

```bash
git clone https://github.com/Lesh79/PySATL-CPD-Module
```

Install dependencies:

```bash
poetry install
```

## CPD module usage example:

```python
# import needed CPD algorithm from pysatl_cpd.core
from pysatl_cpd.core.algorithms.graph_algorithm import GraphAlgorithm
from pysatl_cpd.core.problem import CpdProblem
from pysatl_cpd.core.scrubber.linear import LinearScrubber
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider

# import solver
from pysatl_cpd.cpd_solver import CpdSolver

# specify data scrubber
scrubber = LinearScrubber(ListUnivariateProvider([1] * 100 + [50] * 100 + [100] * 100))
# specify CPD algorithm with parameters
algorithm = GraphAlgorithm(lambda a, b: abs(a - b) < 5, 3)
# make a solver object
solver = CpdSolver(CpdProblem(True), algorithm, scrubber)


# then run algorithm
cpd_results = solver.run()

# print the results
print(cpd_results)
# output:
# Located change points: (100;200)
# Computation time (ms): 0.03

# visualize data with located changepoints
cpd_results.visualize()
```
![example_of_output](assets/exam1.png)

## Development

Install requirements

```bash
poetry install --with dev
```

## Pre-commit

Install pre-commit hooks:

```shell
poetry run pre-commit install
```

Starting manually:

```shell
poetry run pre-commit run --all-files --color always --verbose --show-diff-on-failure
```

## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.
