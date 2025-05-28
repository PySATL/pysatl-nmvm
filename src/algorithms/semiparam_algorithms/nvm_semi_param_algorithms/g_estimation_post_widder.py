import cmath
import functools
import math
from typing import TypedDict, Unpack

import numpy as np
from numpy import _typing

from src.estimators.estimate_result import EstimateResult

MU_DEFAULT_VALUE = 0.1
SIGMA_DEFAULT_VALUE = 1.0
N_DEFAULT_VALUE = 2
X_DATA_DEFAULT_VALUE = [1.0]

class SemiParametricGEstimationPostWidder:
    """Estimation of mixing density function g (xi density function) of NVM mixture represented in classical form Y =
    alpha + mu*xi + sigma*sqrt(xi)*N, where alpha = 0 and mu, sigma are given.

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm

    """

    class ParamsAnnotation(TypedDict, total=False):
        """Class for parameters annotation"""
        mu: float
        sigma: float
        n: int
        x_data: list[float]

    def __init__(self, sample: _typing.NDArray = None, **kwargs: Unpack[ParamsAnnotation]):
        self.sample_at_init = np.array([]) if sample is None else sample

        self.mu, self.sigma, self.n, self.x_data = self._validate_kwargs(**kwargs)
        self._H_cache: dict[int, float] = {}
        self._partial_bell_polynomial_recursive.cache_clear()


    @staticmethod
    def _validate_kwargs(**kwargs: Unpack[ParamsAnnotation]) -> tuple[float, float, int, list[float]]:
        mu = kwargs.get("mu", MU_DEFAULT_VALUE)
        sigma = kwargs.get("sigma", SIGMA_DEFAULT_VALUE)
        n_val = kwargs.get("n", N_DEFAULT_VALUE)
        x_data = kwargs.get("x_data", X_DATA_DEFAULT_VALUE)

        if not isinstance(mu, int|float):
            raise TypeError("mu must be a float or integer.")
        if (not isinstance(sigma, int|float)) or (sigma <= 0):
            raise TypeError("sigma must be a positive float.")
        if (not isinstance(n_val, int)) or n_val <= 0:
            raise TypeError("N must be a positive integer.")

        return float(mu), float(sigma), n_val, [float(x) for x in x_data]


    def _get_H_l(self, l_idx: int) -> float:
        if l_idx in self._H_cache:
            return self._H_cache[l_idx]

        val = 1.0 if l_idx == 1 else self._get_H_l(l_idx - 1) * float(2 * l_idx - 3)
        self._H_cache[l_idx] = val
        return val

    @functools.cache
    def _partial_bell_polynomial_recursive(self, n_bell: int, k_bell: int) -> float:
        if k_bell < 0 or k_bell > n_bell:
            return 0.0
        if n_bell == 0 and k_bell == 0:
            return 1.0
        if n_bell == 0 or k_bell == 0:
            return 0.0

        res = 0.0
        for i in range(1, n_bell - k_bell + 2):
            if (n_bell - 1) < (i - 1) or (i-1) < 0:
                 term_binom = 0.0
            else:
                term_binom = float(math.comb(n_bell - 1, i - 1))

            if term_binom == 0.0:
                continue

            h_i_val = self._get_H_l(i)
            bell_recursive_val = self._partial_bell_polynomial_recursive(n_bell - i, k_bell - 1)
            res += term_binom * h_i_val * bell_recursive_val
        return res

    def _calculate_F_Nk(self, N_val: int, k_val: int) -> float:
        if k_val <= 0 or k_val > N_val:
            return 0.0
        return self._partial_bell_polynomial_recursive(N_val, k_val)

    def algorithm(self, sample: np.ndarray) -> EstimateResult:
        """Estimate g(x)

        Args:
            sample: sample of the analysed distribution

        Returns: estimated g function value in x_data points

        """
        N_val = self.n
        mu_param = self.mu
        sigma_param = self.sigma
        x_data_points = self.x_data

        sample_size = len(sample)
        pw_coeff_part1 = ((-1)**N_val) / math.factorial(N_val)

        estimated_g_values_real = []

        for x_point in x_data_points:

            y0 = N_val / x_point

            z_eval = complex(y0, -mu_param * math.sqrt(2 * y0) / sigma_param)
            term_g_pow_N_plus_1 = z_eval ** (N_val + 1)
            coeff_before_sample_sum = pw_coeff_part1 * term_g_pow_N_plus_1

            sum_over_samples_j = complex(0, 0)

            u_tilde_for_exp = math.sqrt(2 * y0) / sigma_param

            base_for_power_term_in_k_sum = 2 * z_eval - mu_param**2

            for X_j_float in sample:
                exp_term = cmath.exp(complex(0, u_tilde_for_exp * X_j_float))

                sum_over_k_val = complex(0, 0)
                for k_idx in range(1, N_val + 1):
                    F_Nk = self._calculate_F_Nk(N_val, k_idx)
                    if F_Nk == 0.0:
                        continue

                    term_Xj_k = (complex(0, X_j_float)) ** k_idx
                    term_minus_1_pow = (-1)**(N_val - k_idx)

                    power_val = float(k_idx/2.0 - N_val)
                    term_base_pow = base_for_power_term_in_k_sum ** power_val

                    term_in_sum_k = term_Xj_k * term_minus_1_pow * F_Nk * term_base_pow
                    sum_over_k_val += term_in_sum_k

                sum_over_samples_j += exp_term * sum_over_k_val

            p_nx_complex = coeff_before_sample_sum * (sum_over_samples_j / sample_size)
            estimated_g_values_real.append(p_nx_complex.real)

        return EstimateResult(list_value=estimated_g_values_real)
