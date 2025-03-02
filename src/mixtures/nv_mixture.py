from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.rqmc import RQMC
from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NVM"""
    alpha: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


@dataclass
class _NVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NVM"""
    alpha: float | int | np.int64
    distribution: rv_frozen | rv_continuous


class NormalVarianceMixtures(AbstractMixtures):

    _classical_collector = _NVMClassicDataCollector
    _canonical_collector = _NVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def compute_moment(self, n: int, params: dict) -> tuple[float, float]:
        """
        Compute n-th moment of  NVM
        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm
        Returns: moment approximation and error tolerance
        """
        gamma = self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1
        def integrate_func(u: float) -> float:
            return sum([binom(n, k) * (gamma ** k) * (self.params.alpha ** (n - k)) * (self.params.distribution.ppf(u) ** (k/2)) * norm.moment(k) for k in range(0, n + 1)])
        result = RQMC(integrate_func, **params)()
        return result

    def compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        parametric_norm = norm(0, self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1)
        rqmc = RQMC(
            lambda u: parametric_norm.cdf((x - self.params.alpha) / np.sqrt(self.params.distribution.ppf(u))), **params
        )
        return rqmc()

    @lru_cache()
    def _integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return (1 / np.sqrt(np.pi * 2 * ppf * np.abs(gamma**2))) * np.exp(-1 * d / (2 * ppf))

    def _log_integrand_func(self, u: float, d: float, gamma: float | int | np.int64) -> float:
        ppf = self.params.distribution.ppf(u)
        return -(ppf * np.log(np.pi * 2 * ppf * gamma**2) + d) / (2 * ppf)

    def compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        gamma = self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1
        d = (x - self.params.alpha) ** 2 / gamma**2
        rqmc = RQMC(lambda u: self._integrand_func(u, d, gamma), **params)
        return rqmc()

    def compute_logpdf(self, x: float, params: dict) -> tuple[float, float]:
        gamma = self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1
        d = (x - self.params.alpha) ** 2 / gamma**2
        log_rqmc = LogRQMC(lambda u: self._log_integrand_func(u, d, gamma), **params)
        return log_rqmc()
