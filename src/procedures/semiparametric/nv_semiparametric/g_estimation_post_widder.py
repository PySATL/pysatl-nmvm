import math
from typing import TypedDict, Unpack

import mpmath
import numpy as np
from numpy import _typing
from sympy import bell

from src.estimators.estimate_result import EstimateResult

SIGMA_DEFAULT_VALUE = 1.0
N_DEFAULT_VALUE = 2
X_DATA_DEFAULT_VALUE = [1.0]


class NVEstimationDensityPW:
    """Estimation of mixing density function g (xi density function) of NVM mixture represented in classical form Y =
    alpha + sigma*sqrt(xi)*N, where alpha = 0 and mu, sigma are given.

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm

    """

    class ParamsAnnotation(TypedDict, total=False):
        """Class for parameters annotation"""

        sigma: float
        n: int

        x_data: list[float]

    def __init__(self, sample: _typing.NDArray = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample = np.array([]) if sample is None else sample
        self.sample_size = len(self.sample)
        self.sigma, self.n, self.x_data = self._validate_kwargs(**kwargs)
        self.p_x_first_factor = ((-1) ** self.n) / math.factorial(self.n)

    @staticmethod
    def _validate_kwargs(**kwargs: Unpack[ParamsAnnotation]) -> tuple[float, int, list[float]]:
        sigma = kwargs.get("sigma", SIGMA_DEFAULT_VALUE)
        n = kwargs.get("n", N_DEFAULT_VALUE)
        x_data = kwargs.get("x_data", X_DATA_DEFAULT_VALUE)
        return sigma, n, x_data

    def generalized_post_widder_g(self, y: float) -> complex:
        return complex(y, 0)

    def xi(self, z: complex) -> complex:
        return (2 * z) ** 0.5

    def f_coeff(self, k: int) -> int:
        x_last = (math.factorial(2 * (self.n - k))) / (2 ** (self.n - k) * math.factorial(self.n - k))
        return int(bell(self.n, k, range(1, round(x_last) + 1)))

    def p_x_estimation(self, x: float) -> float:
        j_sum = complex(0, 0)
        for j in range(1, self.sample_size + 1):
            k_sum = complex(0, 0)
            for k in range(1, self.n + 1):
                k_sum = (
                    k_sum
                    + complex(0, self.sample[j - 1]) ** k
                    * (-1) ** (self.n - k)
                    * (2 * self.generalized_post_widder_g(self.n / x)) ** (k / 2 - self.n)
                ) * self.f_coeff(k)
            exp_factor = mpmath.exp(
                complex(0, 1) * self.xi(self.generalized_post_widder_g(self.n / x)) * self.sample[j - 1]
            )
            j_sum += exp_factor * k_sum
        result = (
            self.p_x_first_factor
            * self.generalized_post_widder_g(self.n / x) ** (self.n + 1)
            * (1 / self.sample_size)
            * j_sum
        )
        return result.real

    def compute(self, sample: np.ndarray) -> EstimateResult:
        """Estimate g(x)

        Args:
            sample: sample of the analysed distribution

        Returns: estimated g function value in x_data points

        """
        y_data = [max(0.0, self.p_x_estimation(x)) for x in self.x_data]
        return EstimateResult(list_value=y_data, success=True)
