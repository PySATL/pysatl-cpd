"""
Module for Bayesian CPD algorithm's customization blocks.
"""

__author__ = "Alexey Tatyanenko, Loikov Vladislav"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.bayesian.abstracts import (
    IChangePointFunction,
    IHazard,
    ILikelihood,
    ILikelihoodWithPriorProbability,
    ILocalizer,
)
from pysatl_cpd.core.algorithms.bayesian.hazards import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods import (
    ExponentialConjugate,
    ExponentialConjugateWithPriorProbability,
    Gaussian,
    GaussianConjugate,
    GaussianConjugateWithPriorProbability,
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian.test_statistics import (
    DropCPF,
    MaxRunLengthCPF,
)

__all__ = [
    "ArgmaxLocalizer",
    "ConstantHazard",
    "DropCPF",
    "ExponentialConjugate",
    "ExponentialConjugateWithPriorProbability",
    "Gaussian",
    "GaussianConjugate",
    "GaussianConjugateWithPriorProbability",
    "HeuristicGaussianVsExponential",
    "IChangePointFunction",
    "IHazard",
    "ILikelihood",
    "ILikelihoodWithPriorProbability",
    "ILocalizer",
    "MaxRunLengthCPF",
]
