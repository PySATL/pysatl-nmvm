import math
from typing import Optional, Tuple, TypedDict

import numpy as np
from scipy.linalg import eigh

from src.estimators.estimate_result import EstimateResult

L_DEFAULT_VALUE = 5
K_DEFAULT_VALUE = 10
EPS_DEFAULT_VALUE = 0.1
SEARCH_AREA_DEFAULT_VALUE = 10
SEARCH_DENSITY_DEFAULT_VALUE = 1000
PARAMETER_KEYS = ["l", "k", "eps", "search_area", "search_density"]


class SemiParametricMeanSigmaEstimationEigenvalueBased:
    """Estimation of sigma parameter of NM mixture represented in canonical form Y = xi + sigma*N.

    Args:
        sample: Sample of the analyzed distribution.
        params: Parameters of the algorithm.
    """

    class ParamsAnnotation(TypedDict, total=False):
        l: float
        k: float
        eps: float
        search_area: float
        search_density: int

    def __init__(self, sample: Optional[np.ndarray] = None, **kwargs: ParamsAnnotation):
        self.sample: np.ndarray = np.array([]) if sample is None else np.asarray(sample)
        self.n: int = len(self.sample)
        (
            self.l,
            self.k,
            self.eps,
            self.search_area,
            self.search_density,
        ) = self._validate_kwargs(**kwargs)

    @staticmethod
    def _validate_kwargs(**kwargs: ParamsAnnotation) -> Tuple[float, float, float, float, int]:
        """Validate and extract parameters.

        Args:
            kwargs: Parameters of the algorithm.

        Returns:
            Tuple[float, float, float, float, int]: Validated parameters.
        """
        if any(key not in PARAMETER_KEYS for key in kwargs):
            raise ValueError("Got unexpected parameter.")
        l = kwargs.get("l", L_DEFAULT_VALUE)
        k = kwargs.get("k", K_DEFAULT_VALUE)
        eps = kwargs.get("eps", EPS_DEFAULT_VALUE)
        search_area = kwargs.get("search_area", SEARCH_AREA_DEFAULT_VALUE)
        search_density = kwargs.get("search_density", SEARCH_DENSITY_DEFAULT_VALUE)
        if not isinstance(l, float) or l <= 0:
            raise ValueError("Expected positive float as parameter 'l'.")
        if not isinstance(k, float) or k <= 0:
            raise ValueError("Expected positive float as parameter 'k'.")
        if not isinstance(eps, float) or eps <= 0:
            raise ValueError("Expected positive float as parameter 'eps'.")
        if not isinstance(search_area, float) or search_area <= 0:
            raise ValueError("Expected positive float as parameter 'search_area'.")
        if not isinstance(search_density, int) or search_density <= 0:
            raise ValueError("Expected positive integer as parameter 'search_density'.")
        return l, k, eps, search_area, search_density

    def _alpha(self, zeta: float, tau: float) -> complex:
        return (1 / self.n) * np.sum(np.exp(1j * zeta * self.sample)) * math.exp(zeta**2 * tau**2 / 2)

    def _generate_t(self) -> np.ndarray:
        k_values = np.arange(-self.l, self.l + 1)
        return k_values / self.k

    def _build_matrix(self, tau: float) -> np.ndarray:
        """Build the matrix for eigenvalue computation."""
        t = self._generate_t()
        t_len = len(t)
        matrix = np.zeros((t_len, t_len), dtype=np.complex128)
        for i in range(t_len):
            for j in range(t_len):
                matrix[i, j] = self._alpha(t[i] - t[j], tau)
        return matrix

    def compute(self, sample: np.ndarray) -> EstimateResult:
        """Estimate sigma.

        Args:
            sample: Sample of the analyzed distribution.

        Returns:
            EstimateResult: Object with estimated sigma value.
        """
        if sample.size == 0:
            raise ValueError("Sample cannot be empty.")
        tau_values = np.linspace(0, self.search_area, self.search_density)
        for tau in tau_values:
            matrix = self._build_matrix(tau)
            eigenvalues = eigh(matrix, eigvals_only=True)
            lambda_min = np.min(eigenvalues)
            if lambda_min < -self.eps:
                return EstimateResult(value=tau, success=True)
        return EstimateResult(success=False)
