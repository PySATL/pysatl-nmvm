from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Type, Dict, Tuple

import numpy as np
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.integrator import Integrator
from src.algorithms.support_algorithms.rqmc import RQMCIntegrator
from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.mixtures.abstract_mixture import AbstractMixtures

@dataclass
class _NMVMClassicDataCollector:
    alpha: float | int | np.int64
    beta:  float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous

@dataclass
class _NMVMCanonicalDataCollector:
    alpha: float | int | np.int64
    mu:    float | int | np.int64
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

    def _compute_moment(self, n: int) -> Tuple[float, float]:
        gamma = getattr(self.params, 'gamma', None)

        def integrand(u: float) -> float:
            s = 0.0
            for k in range(n + 1):
                for l in range(k + 1):
                    if self.mixture_form == 'classical':
                        term = (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.beta ** (k - l))
                            * (self.params.gamma ** l)
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * norm.moment(l)
                        )
                    else:
                        term = (
                            binom(n, n - k)
                            * binom(k, k - l)
                            * (self.params.mu ** (k - l))
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * norm.moment(l)
                        )
                    s += term
            return s

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error

    def _compute_cdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == 'classical':
                return norm.cdf((x - self.params.alpha) / (np.sqrt(p) * self.params.gamma))
            return norm.cdf((x - self.params.alpha) / np.sqrt(p) - self.params.mu * np.sqrt(p))

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error

    def _compute_pdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == 'classical':
                return (
                    1 / np.sqrt(2 * np.pi * p * self.params.gamma ** 2)
                    * np.exp(-((x - self.params.alpha) ** 2 + self.params.beta ** 2 * p ** 2) / (2 * p * self.params.gamma ** 2))
                )
            return (
                1 / np.sqrt(2 * np.pi * p)
                * np.exp(-((x - self.params.alpha) ** 2 + self.params.mu ** 2 * p ** 2) / (2 * p))
            )

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        if self.mixture_form == 'classical':
            val = np.exp(self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2) * res.value
        else:
            val = np.exp(self.params.mu * (x - self.params.alpha)) * res.value
        return val, res.error

    def _compute_logpdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == 'classical':
                return -((x - self.params.alpha) ** 2 + p ** 2 * self.params.beta ** 2 + p * self.params.gamma ** 2 * np.log(2 * np.pi * p * self.params.gamma ** 2)) / (2 * p * self.params.gamma ** 2)
            return -((x - self.params.alpha) ** 2 + p ** 2 * self.params.mu ** 2 + p * np.log(2 * np.pi * p)) / (2 * p)

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        if self.mixture_form == 'classical':
            val = self.params.beta * (x - self.params.alpha) / self.params.gamma ** 2 + res.value
        else:
            val = self.params.mu * (x - self.params.alpha) + res.value
        return val, res.error
