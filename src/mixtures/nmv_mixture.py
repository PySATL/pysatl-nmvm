from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Type, Dict

import numpy as np
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.integrator import Integrator
from src.algorithms.support_algorithms.rqmc import RQMCIntegrator
from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.quad_integrator import QuadIntegrator
from src.mixtures.abstract_mixture import AbstractMixtures

@dataclass
class _NMVMClassicDataCollector:
    alpha: float | int | np.int64
    beta: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous

@dataclass
class _NMVMCanonicalDataCollector:
    alpha: float | int | np.int64
    mu: float | int | np.int64
    distribution: rv_frozen | rv_continuous

class NormalMeanVarianceMixtures(AbstractMixtures):
    _classical_collector = _NMVMClassicDataCollector
    _canonical_collector = _NMVMCanonicalDataCollector

    def __init__(
        self,
        mixture_form: str,
        integrator_cls: Type[Integrator] = RQMCIntegrator,
        integrator_params: Dict[str, Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(mixture_form, integrator_cls=integrator_cls, integrator_params=integrator_params, **kwargs)

    def _compute_moment(self, n: int) -> tuple[float, float]:
        def integrand(u: float) -> float:
            result = 0.0
            for k in range(n + 1):
                for l in range(k + 1):
                    if self.mixture_form == "classical":
                        result += (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.beta ** (k - l))
                            * (self.params.gamma ** l)
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * norm.moment(l)
                        )
                    else:
                        result += (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.mu ** (k - l))
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * norm.moment(l)
                        )
            return result

        integrator = self.integrator_cls(**(self.integrator_params or {}))
        result = integrator.compute(integrand)
        return result.value, result.error

    def _compute_cdf(self, x: float) -> tuple[float, float]:
        def integrand(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                point = (x - self.params.alpha) / (np.sqrt(ppf) * self.params.gamma) - (self.params.beta / self.params.gamma * np.sqrt(ppf))
            else:
                point = (x - self.params.alpha) / np.sqrt(ppf) - (self.params.mu * np.sqrt(ppf))
            return norm.cdf(point)

        integrator = self.integrator_cls(**(self.integrator_params or {}))
        result = integrator.compute(integrand)
        return result.value, result.error

    def _compute_pdf(self, x: float) -> tuple[float, float]:
        def integrand(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return (
                    1 / np.sqrt(2 * np.pi * ppf * self.params.gamma ** 2)
                    * np.exp(-((x - self.params.alpha) ** 2 + self.params.beta ** 2 * ppf ** 2) / (2 * ppf * self.params.gamma ** 2))
                )
            else:
                return (
                    1 / np.sqrt(2 * np.pi * ppf)
                    * np.exp(-((x - self.params.alpha) ** 2 + self.params.mu ** 2 * ppf ** 2) / (2 * ppf))
                )

        integrator = self.integrator_cls(**(self.integrator_params or {}))
        result = integrator.compute(integrand)
        if self.mixture_form == "classical":
            val = np.exp(self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2) * result.value
        else:
            val = np.exp(self.params.mu * (x - self.params.alpha)) * result.value
        return val, result.error

    def _compute_logpdf(self, x: float) -> tuple[float, float]:
        def integrand(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return -(
                    (x - self.params.alpha) ** 2
                    + ppf ** 2 * self.params.beta ** 2
                    + ppf * self.params.gamma ** 2 * np.log(2 * np.pi * ppf * self.params.gamma ** 2)
                ) / (2 * ppf * self.params.gamma ** 2)
            else:
                return -((x - self.params.alpha) ** 2 + ppf ** 2 * self.params.mu ** 2 + ppf * np.log(2 * np.pi * ppf)) / (2 * ppf)

        integrator = self.integrator_cls(**(self.integrator_params or {}))
        result = integrator.compute(integrand)
        if self.mixture_form == "classical":
            val = self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2 + result.value
        else:
            val = self.params.mu * (x - self.params.alpha) + result.value
        return val, result.error

    @lru_cache()
    def _integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return (1 / np.sqrt(2 * np.pi * ppf * abs(gamma) ** 2)) * np.exp(-d / (2 * ppf))
