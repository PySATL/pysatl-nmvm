"""Empirical sigma estimation for NM semiparametric mixtures.

This module implements empirical sigma estimation methods for
Normal Mean semiparametric mixtures.
"""

import math
from typing import Optional, TypedDict

import numpy as np

from estimators.estimate_result import EstimateResult

T_DEFAULT_VALUE = 7.5
PARAMETER_KEYS = ["t"]


class SemiParametricMeanSigmaEstimationEmpirical:
    """Estimation of sigma parameter of NM mixture represented in canonical form Y = xi + sigma*N.

    Args:
        sample (Optional[np.ndarray]): Sample of the analyzed distribution.
        params (Optional[dict]): Parameters of the algorithm.
    """

    class ParamsAnnotation(TypedDict, total=False):
        t: float

    def __init__(
        self,
        sample: Optional[np.ndarray] = None,
        **kwargs: ParamsAnnotation,
    ):
        self.sample: np.ndarray = np.array([]) if sample is None else sample
        self.n: int = len(self.sample)
        self.t: float = self._validate_kwargs(**kwargs)

    @staticmethod
    def _validate_kwargs(**kwargs: ParamsAnnotation) -> float:
        """Validate and extract parameters.

        Args:
            kwargs (dict): Parameters of the algorithm.

        Returns:
            float: Validated parameter `t`.
        """
        if any(key not in PARAMETER_KEYS for key in kwargs):
            raise ValueError("Got unexpected parameter.")

        t_value = kwargs.get("t", T_DEFAULT_VALUE)
        if not isinstance(t_value, (float, int)):
            raise ValueError("Expected a numeric value (float or int) as parameter `t`.")

        t = float(t_value)
        if t <= 0:
            raise ValueError("Expected a positive float as parameter `t`.")
        return t

    def algorithm(self, sample: np.ndarray) -> EstimateResult:
        """Estimate the sigma parameter.

        Args:
            sample (np.ndarray): Sample of the analyzed distribution.

        Returns:
            EstimateResult: An object containing the estimated sigma value.
        """
        if sample.size == 0:
            raise ValueError("Sample can't be empty.")
        sigma = ((2 / (self.t**2)) * math.log((1 / self.n) * sum([math.exp(self.t * x) for x in self.sample]))) ** 0.5
        return EstimateResult(value=sigma, success=True)
