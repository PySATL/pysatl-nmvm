import math
from typing import Callable, Optional, TypedDict, Unpack

import mpmath
import numpy as np
from numpy import _typing

from src.estimators.estimate_result import EstimateResult

M_DEFAULT_VALUE = 1000
TOLERANCE_DEFAULT_VALUE = 10**-5
OMEGA_DEFAULT_VALUE = lambda x: -1 * math.sin(x) if abs(x) <= math.pi else 0
MAX_ITERATIONS_DEFAULT_VALUE = 10**9
PARAMETER_KEYS = ["m", "tolerance", "omega", "max_iterations"]


class NMVEstimationMu:
    """Estimation of mu parameter of NVM mixture represented in canonical form Y = alpha + mu*xi + sqrt(xi)*N,
    where alpha = 0

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm
                m - search area,
                tolerance - defines where to stop binary search,
                omega - Lipschitz continuous odd function on R with compact support

    """

    class ParamsAnnotation(TypedDict, total=False):
        """Class for parameters annotation"""

        m: float
        tolerance: float
        omega: Callable[[float], float]
        max_iterations: float

    def __init__(self, sample: Optional[_typing.ArrayLike] = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample = np.array([]) if sample is None else sample
        self.m, self.tolerance, self.omega, self.max_iterations = self._validate_kwargs(**kwargs)

    @staticmethod
    def _validate_kwargs(**kwargs: Unpack[ParamsAnnotation]) -> tuple[float, float, Callable[[float], float], float]:
        """Parameters validation function

        Args:
            kwargs: Parameters of Algorithm

        Returns: Parameters of Algorithm

        """
        if any([i not in PARAMETER_KEYS for i in kwargs]):
            raise ValueError("Got unexpected parameter")
        m = kwargs.get("m", M_DEFAULT_VALUE)
        tolerance = kwargs.get("tolerance", TOLERANCE_DEFAULT_VALUE)
        omega = kwargs.get("omega", OMEGA_DEFAULT_VALUE)
        max_iterations = kwargs.get("max_iterations", MAX_ITERATIONS_DEFAULT_VALUE)
        if not isinstance(m, int) or m <= 0:
            raise ValueError("Expected positive integer as parameter m")
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError("Expected positive float as parameter tolerance")
        if not callable(omega):
            raise ValueError("Expected callable object as parameter omega")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("Expected positive integer as parameter max_iterations")
        return m, tolerance, omega, max_iterations

    def __w(self, p: float, sample: np._typing.NDArray) -> float:
        """Root of this function is an estimation of mu

        Args:
            p: float
            sample: sample of the analysed distribution

        Returns: function value

        """
        y = 0.0
        for x in sample:
            try:
                e = math.exp(-p * x)
            except OverflowError:
                e = mpmath.exp(-p * x)
            y += e * self.omega(x)
        return y

    def compute(self, sample: np._typing.NDArray) -> EstimateResult:
        """Root of this function is an estimation of mu

        Args:
            sample: sample of the analysed distribution

        Returns: estimated mu value

        """
        if self.__w(0, sample) == 0:
            return EstimateResult(value=0, success=True)
        if self.__w(0, sample) > 0:
            second_result = self.compute(-1 * sample)
            return EstimateResult(value=-1 * second_result.value, success=second_result.success)
        if self.__w(self.m, sample) < 0:
            return EstimateResult(value=self.m, success=False)

        left, right = 0.0, self.m
        iteration = 0
        while left <= right:
            mid = (right + left) / 2
            if iteration > self.max_iterations:
                return EstimateResult(value=mid, success=False)
            iteration += 1
            if abs(self.__w(mid, sample)) < self.tolerance:
                return EstimateResult(value=mid, success=True)
            elif self.__w(mid, sample) < 0:
                left = mid
            else:
                right = mid
        return EstimateResult(value=-1, success=False)
