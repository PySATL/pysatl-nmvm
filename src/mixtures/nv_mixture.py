from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
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

    def _compute_moment(self, n: int, rqmc_params: dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)

        def integrand(u: float) -> float:
            return sum(
                binom(n, k)
                * (gamma ** k)
                * (self.params.alpha ** (n - k))
                * (self.params.distribution.ppf(u) ** (k / 2))
                * norm.moment(k)
                for k in range(n + 1)
            )

        return RQMC(integrand, **rqmc_params)()

    def _compute_cdf(self, x: float, rqmc_params: dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        param_norm = norm(0, gamma)

        def integrand(u: float) -> float:
            return param_norm.cdf((x - self.params.alpha) / np.sqrt(self.params.distribution.ppf(u)))

        return RQMC(integrand, **rqmc_params)()


    def _compute_pdf(self, x: float, rqmc_params: dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        d = (x - self.params.alpha) ** 2 / gamma ** 2

        def integrand(u: float) -> float:
            return self._integrand_func(u, d, gamma)

        return RQMC(integrand, **rqmc_params)()

    def _compute_logpdf(self, x: float, rqmc_params: dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        d = (x - self.params.alpha) ** 2 / gamma ** 2

        def integrand(u: float) -> float:
            return self._log_integrand_func(u, d, gamma)

        return LogRQMC(integrand, **rqmc_params)()

    @lru_cache()
    def _integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return (1 / np.sqrt(np.pi * 2 * ppf * np.abs(gamma ** 2))) * np.exp(-d / (2 * ppf))

    def _log_integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return -(ppf * np.log(np.pi * 2 * ppf * gamma ** 2) + d) / (2 * ppf)
