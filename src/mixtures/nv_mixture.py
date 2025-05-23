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


def compute_moment(self, ns: list[int], params: dict) -> list[tuple[float, float]]:
    gamma = getattr(self.params, 'gamma', 1)

    def make_integrand(n):
        def func(u):
            return sum(
                binom(n, k)
                * (gamma ** k)
                * (self.params.alpha ** (n - k))
                * (self.params.distribution.ppf(u) ** (k / 2))
                * norm.moment(k)
                for k in range(n + 1)
            )

        return func

    return [RQMC(make_integrand(n), **params)() for n in ns]


def compute_cdf(self, xs: list[float], params: dict) -> list[tuple[float, float]]:
    gamma = getattr(self.params, 'gamma', 1)
    param_norm = norm(0, gamma)

    def make_cdf_integrand(x):
        def func(u):
            return param_norm.cdf((x - self.params.alpha) / np.sqrt(self.params.distribution.ppf(u)))

        return func

    return [RQMC(make_cdf_integrand(x), **params)() for x in xs]


def compute_pdf(self, xs: list[float], params: dict) -> list[tuple[float, float]]:
    gamma = getattr(self.params, 'gamma', 1)

    def make_pdf_integrand(d):
        return lambda u: self._integrand_func(u, d, gamma)

    return [RQMC(make_pdf_integrand((x - self.params.alpha) ** 2 / gamma ** 2), **params)() for x in xs]


def compute_logpdf(self, xs: list[float], params: dict) -> list[tuple[float, float]]:
    gamma = getattr(self.params, 'gamma', 1)

    def make_log_integrand(d):
        return lambda u: self._log_integrand_func(u, d, gamma)

    return [LogRQMC(make_log_integrand((x - self.params.alpha) ** 2 / gamma ** 2), **params)() for x in xs]


@lru_cache()
def _integrand_func(self, u: float, d: float, gamma: float) -> float:
    ppf = self.params.distribution.ppf(u)
    return (1 / np.sqrt(np.pi * 2 * ppf * np.abs(gamma ** 2))) * np.exp(-d / (2 * ppf))


def _log_integrand_func(self, u: float, d: float, gamma: float) -> float:
    ppf = self.params.distribution.ppf(u)
    return -(ppf * np.log(np.pi * 2 * ppf * gamma ** 2) + d) / (2 * ppf)