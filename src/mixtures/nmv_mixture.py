from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import binom
from scipy.stats import geninvgauss, norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.integrator import Integrator
from src.algorithms.support_algorithms.rqmc import RQMCIntegrator
from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NMVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NMVM"""
    alpha: float | int | np.int64
    beta: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NMVM"""
    alpha: float | int | np.int64
    mu: float | int | np.int64
    distribution: rv_frozen | rv_continuous


class NormalMeanVarianceMixtures(AbstractMixtures):
    _classical_collector = _NMVMClassicDataCollector
    _canonical_collector = _NMVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        self.mixture_form = mixture_form
        super().__init__(mixture_form, **kwargs)

    def _compute_moment(self, n: int, params: dict) -> tuple[float, float]:
        def integral_func(u: float) -> float:
            result = 0
            for k in range(0, n + 1):
                for l in range(0, k + 1):
                    if self.mixture_form == "classical":
                        result += (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.beta ** (k - l))
                            * (self.params.gamma ** l)
                            * self.params.distribution.ppf(u) ** (k - l / 2)
                            * (self.params.alpha ** (n - k))
                            * norm.moment(l)
                        )
                    else:
                        result += (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.nu ** (k - l))
                            * self.params.distribution.ppf(u) ** (k - l / 2)
                            * (self.params.alpha ** (n - k))
                            * norm.moment(l)
                        )
            return result

        rqmc = RQMC(integral_func)
        return rqmc()


    def _compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        def inner_func(u: float) -> float:

            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                point = (x - self.params.alpha) / (np.sqrt(ppf) * self.params.gamma) - (
                    self.params.beta / self.params.gamma * np.sqrt(ppf)
                )
            else:
                point = (x - self.params.alpha) / (np.sqrt(ppf)) - (self.params.mu * np.sqrt(ppf))
            return norm.cdf(point)

        rqmc = RQMC(inner_func)
        return rqmc()


    def _compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        def inner_func(u: float) -> float:

            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return (
                    1
                    / np.sqrt(2 * np.pi * ppf * self.params.gamma ** 2)
                    * np.exp(
                        -((x - self.params.alpha) ** 2 + self.params.beta ** 2 * ppf ** 2)
                        / (2 * ppf * self.params.gamma ** 2)
                    )
                )
            else:
                return (
                    1
                    / np.sqrt(2 * np.pi * ppf)
                    * np.exp(-((x - self.params.alpha) ** 2 + self.params.mu ** 2 * ppf ** 2) / (2 * ppf))
                )

        rqmc_res = RQMC(inner_func)()
        if self.mixture_form == "classical":
            val = np.exp(self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2) * rqmc_res[0]
        else:
            val = np.exp(self.params.mu * (x - self.params.alpha)) * rqmc_res[0]

        return val, rqmc_res[1]


    def _compute_logpdf(self, x: float, params: dict) -> tuple[float, float]:
        def inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return -(
                    (x - self.params.alpha) ** 2
                    + ppf ** 2 * self.params.beta ** 2
                    + ppf * self.params.gamma ** 2 * np.log(2 * np.pi * ppf * self.params.gamma ** 2)
                ) / (2 * ppf * self.params.gamma ** 2)
            else:
                return -((x - self.params.alpha) ** 2 + ppf ** 2 * self.params.mu ** 2 + ppf * np.log(2 * np.pi * ppf)) / (2 * ppf)

        rqmc_res = LogRQMC(inner_func)()
        if self.mixture_form == "classical":
            val = self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2 + rqmc_res[0]
        else:
            val = self.params.mu * (x - self.params.alpha) + rqmc_res[0]

        return val, rqmc_res[1]

