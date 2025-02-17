import math
from typing import TypedDict, Unpack

import mpmath
import numpy as np
import scipy
from numpy import _typing
from scipy.integrate import quad

from src.estimators.estimate_result import EstimateResult

MU_DEFAULT_VALUE = 1
GAMMA_DEFAULT_VALUE = 0.25
U_SEQUENCE_DEFAULT_VALUE = lambda n: n**0.25
V_SEQUENCE_DEFAULT_VALUE = lambda n: math.log(n)
X_DATA_DEFAULT_VALUE = [1.0]


class SemiParametricGEstimationGivenMu:
    """Estimation of mixing density function g (xi density function) of NVM mixture represented in canonical form Y =
    alpha + mu*xi + sqrt(xi)*N, where alpha = 0 and mu is given.

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm

    """

    class ParamsAnnotation(TypedDict, total=False):
        """Class for parameters annotation"""

        mu: float
        gmm: float
        u_value: float
        v_value: float
        x_data: list[float]

    def __init__(self, sample: _typing.NDArray = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample = np.array([]) if sample is None else sample
        self.n = len(self.sample)
        self.mu, self.gmm, self.u_value, self.v_value, self.x_data = self._validate_kwargs(self.n, **kwargs)

    @staticmethod
    def _validate_kwargs(n: int, **kwargs: Unpack[ParamsAnnotation]) -> tuple[float, float, float, float, list[float]]:
        mu = kwargs.get("mu", MU_DEFAULT_VALUE)
        gmm = kwargs.get("gmm", GAMMA_DEFAULT_VALUE)
        u_value = kwargs.get("u_value", U_SEQUENCE_DEFAULT_VALUE(n))
        v_value = kwargs.get("v_value", V_SEQUENCE_DEFAULT_VALUE(n))
        x_data = kwargs.get("x_data", X_DATA_DEFAULT_VALUE)
        return mu, gmm, u_value, v_value, x_data

    def conjugate_psi(self, u: float) -> complex:
        return complex((u**2) / 2, self.mu * u)

    def psi(self, u: float) -> complex:
        return complex((u**2) / 2, -1 * self.mu * u)

    def first_u_integrand(self, u: float, v: float, k: int) -> complex:
        expon_factor = mpmath.exp(complex(0, -1 * u * self.sample[k]))
        conjugate_psi_factor = self.conjugate_psi(u) ** (complex(-self.gmm, -v))
        conjugate_derivative_psi_factor = complex(u, self.mu)
        return expon_factor * conjugate_psi_factor * conjugate_derivative_psi_factor

    def first_v_integrand(self, v: float, k: int, x: float) -> complex:
        return (
            quad(lambda u: self.first_u_integrand(u, v, k), 0, self.u_value, complex_func=True)[0]
            * (x ** complex(-self.gmm, -v))
            / scipy.special.gamma(complex(1 - self.gmm, -v))
        )

    def first_v_integration(self, k: int, x: float) -> complex:
        return quad(lambda v: self.first_v_integrand(v, k, x), 0, self.v_value, complex_func=True)[0]

    def second_u_integrand(self, u: float, v: float, k: int) -> complex:
        expon_factor = mpmath.exp(complex(0, u * self.sample[k]))
        psi_factor = self.psi(u) ** (complex(-self.gmm, -v))
        derivative_psi_factor = complex(u, -self.mu)
        return expon_factor * psi_factor * derivative_psi_factor

    def second_v_integrand(self, v: float, k: int, x: float) -> complex:
        return (
            quad(lambda u: self.second_u_integrand(u, v, k), 0, self.u_value, complex_func=True)[0]
            * (x ** complex(-self.gmm, -v))
            / scipy.special.gamma(complex(1 - self.gmm, -v))
        )

    def second_v_integration(self, k: int, x: float) -> complex:
        return quad(lambda v: self.second_v_integrand(v, k, x), -self.v_value, 0, complex_func=True)[0]

    def g_x_estimation(self, x: float) -> float:
        func_value = complex(0)
        for k in range(self.n):
            func_value += (self.first_v_integration(k, x) + self.second_v_integration(k, x)) / (2 * math.pi * self.n)
        return func_value.real

    def algorithm(self, sample: np._typing.NDArray) -> EstimateResult:
        """Estimate g for given mu.

        Args:
            sample: sample of the analysed distribution

        Returns: estimated g function value in x_data points

        """
        y_data = [max(0, self.g_x_estimation(x)) for x in self.x_data]

        return EstimateResult(list_value=y_data, success=True)
