import math
from bisect import bisect_left
from typing import Callable, Dict, List, Optional, TypedDict, Unpack

import numpy as np
from numpy import _typing
from scipy.integrate import quad_vec
from scipy.special import gamma

from src.estimators.estimate_result import EstimateResult

GAMMA_DEFAULT_VALUE = 0.25


def u_sequence_default_value(n: float) -> float:
    """Default sequence for u: n^0.25."""
    return n**0.25


def v_sequence_default_value(n: float) -> float:
    """Default sequence for v: log(n)."""
    return math.log(n)


U_SEQUENCE_DEFAULT_VALUE: Callable[[float], float] = u_sequence_default_value
V_SEQUENCE_DEFAULT_VALUE: Callable[[float], float] = v_sequence_default_value
X_DATA_DEFAULT_VALUE: List[float] = [1.0]
GRID_SIZE_DEFAULT_VALUE: int = 200
INTEGRATION_TOLERANCE_DEFAULT_VALUE: float = 1e-2
INTEGRATION_LIMIT_DEFAULT_VALUE: int = 50


class NVEstimationDensityInvMT:
    """Estimation of mixing density function g (xi density function) of NV mixture represented in canonical form Y =
    alpha + sqrt(xi)*N, where alpha = 0 and mu = 0.

    Args:
        sample: sample of the analysed distribution
        params: parameters of the algorithm

    """

    class ParamsAnnotation(TypedDict, total=False):
        gmm: float
        u_value: float
        v_value: float
        x_data: List[float]
        grid_size: int
        integration_tolerance: float
        integration_limit: int

    def __init__(self, sample: Optional[np.ndarray] = None, **kwargs: Unpack[ParamsAnnotation]):
        self.x_powers: Dict[float, np.ndarray] = {}
        self.second_u_integrals: np.ndarray
        self.first_u_integrals: np.ndarray
        self.gamma_grid: np.ndarray
        self.v_grid: np.ndarray
        self.sample: np.ndarray = np.array([]) if sample is None else sample
        self.n: int = len(self.sample)
        (
            self.gmm,
            self.u_value,
            self.v_value,
            self.x_data,
            self.grid_size,
            self.integration_tolerance,
            self.integration_limit,
        ) = self._validate_kwargs(self.n, **kwargs)
        self.denominator: float = 2 * math.pi * self.n
        self.precompute_gamma_grid()
        self.precompute_x_powers()
        self.precompute_u_integrals()

    @staticmethod
    def _validate_kwargs(
        n: int, **kwargs: Unpack[ParamsAnnotation]
    ) -> tuple[float, float, float, List[float], int, float, int]:
        gmm: float = kwargs.get("gmm", GAMMA_DEFAULT_VALUE)
        u_value: float = kwargs.get("u_value", U_SEQUENCE_DEFAULT_VALUE(n))
        v_value: float = kwargs.get("v_value", V_SEQUENCE_DEFAULT_VALUE(n))
        x_data: List[float] = kwargs.get("x_data", X_DATA_DEFAULT_VALUE)
        grid_size: int = kwargs.get("grid_size", GRID_SIZE_DEFAULT_VALUE)
        integration_tolerance: float = kwargs.get("integration_tolerance", INTEGRATION_TOLERANCE_DEFAULT_VALUE)
        integration_limit: int = kwargs.get("integration_limit", INTEGRATION_LIMIT_DEFAULT_VALUE)
        return gmm, u_value, v_value, x_data, grid_size, integration_tolerance, integration_limit

    def conjugate_psi(self, u: float) -> complex:
        return complex((u**2) / 2, 0)  # mu = 0

    def psi(self, u: float) -> complex:
        return complex((u**2) / 2, 0)  # mu = 0

    def precompute_gamma_grid(self) -> None:
        self.v_grid = np.linspace(-self.v_value, self.v_value, self.grid_size)  # Symmetric grid
        gamma_vectorized = np.vectorize(lambda v: gamma(complex(1 - self.gmm, -v)))
        self.gamma_grid = gamma_vectorized(self.v_grid)

    def precompute_x_powers(self) -> None:
        exponents = -self.gmm - 1j * self.v_grid

        for x in self.x_data:
            self.x_powers[x] = np.power(x, exponents)

    def first_u_integrand(self, u: float, v: float) -> np.ndarray:
        expon_factor = np.exp(-1j * u * self.sample)
        conjugate_psi_factor = self.conjugate_psi(u) ** complex(-self.gmm, -v)
        return expon_factor * conjugate_psi_factor

    def second_u_integrand(self, u: float, v: float) -> np.ndarray:
        expon_factor = np.exp(1j * u * self.sample)
        psi_factor = self.psi(u) ** complex(-self.gmm, -v)
        return expon_factor * psi_factor

    def precompute_u_integrals(self) -> None:
        self.first_u_integrals = np.zeros((self.grid_size, self.n), dtype=np.complex_)
        self.second_u_integrals = np.zeros((self.grid_size, self.n), dtype=np.complex_)

        for i, v in enumerate(self.v_grid):
            self.first_u_integrals[i] = quad_vec(
                lambda u: self.first_u_integrand(u, v),
                0,
                self.u_value,
                epsabs=self.integration_tolerance,
                limit=self.integration_limit,
            )[0]

            self.second_u_integrals[i] = quad_vec(
                lambda u: self.second_u_integrand(u, v),
                0,
                self.u_value,
                epsabs=self.integration_tolerance,
                limit=self.integration_limit,
            )[0]

    @staticmethod
    def find_closest_index(grid: np.ndarray, value: float) -> int:
        idx = bisect_left(grid, value)
        if idx == 0:
            return 0
        if idx == len(grid):
            return len(grid) - 1
        before = grid[idx - 1]
        after = grid[idx]
        return idx - 1 if (value - before) < (after - value) else idx

    def first_v_integrand(self, v: float, x: float) -> np.ndarray:
        idx = self.find_closest_index(self.v_grid, v)
        gamma_val = self.gamma_grid[idx]
        x_power = self.x_powers[x][idx]
        return (self.first_u_integrals[idx] * x_power) / gamma_val

    def second_v_integrand(self, v: float, x: float) -> np.ndarray:
        idx = self.find_closest_index(self.v_grid, v)
        gamma_val = self.gamma_grid[idx]
        x_power = self.x_powers[x][idx]
        return (self.second_u_integrals[idx] * x_power) / gamma_val

    def compute_integrals_for_x(self, x: float) -> float:
        first_integral = quad_vec(
            lambda v: self.first_v_integrand(v, x),
            0,
            self.v_value,
            epsabs=self.integration_tolerance,
            limit=self.integration_limit,
        )[0]
        second_integral = quad_vec(
            lambda v: self.second_v_integrand(v, x),
            -self.v_value,
            0,
            epsabs=self.integration_tolerance,
            limit=self.integration_limit,
        )[0]
        total = np.sum(first_integral + second_integral) / self.denominator
        return max(0.0, total.real)

    def compute(self, sample: np.ndarray) -> EstimateResult:
        y_data = [self.compute_integrals_for_x(x) for x in self.x_data]
        return EstimateResult(list_value=y_data, success=True)
