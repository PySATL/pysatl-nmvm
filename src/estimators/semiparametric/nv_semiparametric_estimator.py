"""Normal Variance semiparametric estimator module.

This module provides the NVSemiParametricEstimator class for
semiparametric estimation in Normal Variance mixtures.
"""

from numpy import _typing

from estimators.estimate_result import EstimateResult
from estimators.semiparametric.abstract_semiparametric_estimator import AbstractSemiParametricEstimator
from register.algorithm_purpose import AlgorithmPurpose


class NVSemiParametricEstimator(AbstractSemiParametricEstimator):
    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        super().__init__(algorithm_name, params)
        self._purpose = AlgorithmPurpose.NV_SEMIPARAMETRIC

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        return super().estimate(sample)
