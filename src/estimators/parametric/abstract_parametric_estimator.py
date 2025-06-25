"""Abstract parametric estimator base class module.

This module defines the AbstractParametricEstimator base class
for parametric estimation algorithms.
"""

from numpy import _typing

from estimators.abstract_estimator import AbstractEstimator
from estimators.estimate_result import EstimateResult


class AbstractParametricEstimator(AbstractEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
