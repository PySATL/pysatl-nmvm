import math
from typing import Any, Callable, Dict, List, Optional, TypedDict, Unpack

import numpy as np
from numpy import _typing

from src.estimators.estimate_result import EstimateResult

SIGMA_DEFAULT_VALUE: float = 1
BOHMAN_N_DEFAULT_VALUE: int = 10000
BOHMAN_DELTA_DEFAULT_VALUE: float = 0.0001
X_DATA_DEFAULT_VALUE: List[float] = [1.0]


class NMSemiParametricGEstimation:
    """Estimation of mixing density function g (xi density function) of NM mixture
    represented in canonical form Y = xi + sigma*N.

    Args:
        sample: Sample data from the analyzed distribution
        params: Algorithm parameters including:
            - x_data: Evaluation points for density estimation
            - sigman
            - bohman_n
            - bohman_delta

    Raises:
        ValueError: If input sample is empty or invalid parameters provided
    """

    class ParamsAnnotation(TypedDict, total=False):
        x_data: List[float]
        sigma: float
        bohman_n: int
        bohman_delta: float

    def __init__(self, sample: Optional[_typing.NDArray[np.float64]] = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample: _typing.NDArray[np.float64] = np.array([]) if sample is None else sample
        self.n: int = len(self.sample)
        (
            self.x_data,
            self.sigma,
            self.bohman_n,
            self.bohman_delta,
        ) = self._validate_kwargs(**kwargs)

    @staticmethod
    def _validate_kwargs(**kwargs: Unpack[ParamsAnnotation]) -> tuple[List[float], float, int, float]:
        x_data: List[float] = kwargs.get("x_data", X_DATA_DEFAULT_VALUE)
        sigma: float = kwargs.get("sigma", SIGMA_DEFAULT_VALUE)
        bohman_n: int = kwargs.get("bohman_n", BOHMAN_N_DEFAULT_VALUE)
        bohman_delta: float = kwargs.get("bohman_delta", BOHMAN_DELTA_DEFAULT_VALUE)
        return x_data, sigma, bohman_n, bohman_delta

    def characteristic_function_mixture(self, t: float) -> complex:
        smm = 0
        for i in range(self.n):
            smm += np.exp(1j * t * self.sample[i])
        return smm / self.n

    def characteristic_function_normal(self, t: float) -> complex:
        return np.exp(-0.5 * (self.sigma**2) * t**2)

    def characteristic_function_xi(self, t: float) -> complex:
        denominator = np.maximum(np.abs(self.characteristic_function_normal(self.sigma * t)), 1e-10)
        return self.characteristic_function_mixture(t) / denominator

    class BohmanA:
        def __init__(self, N: int = int(1e3), delta: float = 1e-1) -> None:
            super().__init__()
            self.N: int = N
            self.delta: float = delta
            self.coeff_0: float = 0.5
            self.coeff_1: float = 0.0
            self.coeff: np.ndarray = np.array([])

        def fit(self, phi: Callable) -> None:
            self.coeff_0 = 0.5
            self.coeff_1 = self.delta / (2 * np.pi)

            v_values = np.arange(1 - self.N, self.N)
            v_values = v_values[v_values != 0]

            self.coeff = phi(self.delta * v_values) / (2 * np.pi * 1j * v_values)

        def cdf(self, X: np.ndarray) -> np.ndarray:
            v = np.arange(1 - self.N, self.N)
            v_non_zero = v[v != 0]

            x_vect = np.outer(X, v_non_zero)

            F_x = self.coeff_0 + X * self.coeff_1 + (-np.exp(-1j * self.delta * x_vect) @ self.coeff)

            return F_x.real

    def compute(self, sample: _typing.NDArray[np.float64]) -> EstimateResult:
        inv = self.BohmanA(N=self.bohman_n, delta=self.bohman_delta)
        inv.fit(self.characteristic_function_xi)
        x_data_array = np.array(self.x_data, dtype=np.float64)
        estimated_cdf = inv.cdf(x_data_array)
        estimated_pdf = np.gradient(estimated_cdf, x_data_array)
        return EstimateResult(list_value=estimated_pdf.tolist(), success=True)
