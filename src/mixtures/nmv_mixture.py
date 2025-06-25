from dataclasses import dataclass
from typing import Any, Type, Dict, Tuple

import numpy as np
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.integrator import Integrator
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
        integrator_cls: Type[Integrator],
        integrator_params: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(mixture_form, integrator_cls=integrator_cls, integrator_params=integrator_params, **kwargs)

    def _compute_moment(self, n: int) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            s = 0.0
            for k in range(n + 1):
                for l in range(k + 1):
                    if self.mixture_form == "classical":
                        coef = binom(n, n - k) * binom(k, k - l)
                        term = (
                            (self.params.beta ** (k - l))
                            * (self.params.gamma ** l)
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * (self.params.alpha ** (n - k))
                            * norm.moment(l)
                        )
                    else:
                        coef = binom(n, n - k) * binom(k, k - l)
                        term = (
                            (self.params.nu ** (k - l))
                            * (self.params.distribution.ppf(u) ** (k - l/2))
                            * (self.params.alpha ** (n - k))
                            * norm.moment(l)
                        )
                    s += coef * term if self.mixture_form == "classical" else term
            return s

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error

    def _compute_cdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return norm.cdf((x - self.params.alpha - self.params.beta * p) / abs(self.params.gamma))
            return norm.cdf((x - self.params.alpha) / np.sqrt(p) - self.params.mu * np.sqrt(p))

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error

    def _compute_pdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return (1/abs(self.params.gamma)) * norm.pdf((x - self.params.alpha - self.params.beta * p)/abs(self.params.gamma))
            return (1/abs(self.params.mu)) * norm.pdf((x - p)/abs(self.params.mu))

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error

    def _compute_logpdf(self, x: float) -> Tuple[float, float]:
        def integrand(u: float) -> float:
            p = self.params.distribution.ppf(u)
            if self.mixture_form == "classical":
                return np.log(1/abs(self.params.gamma)) + norm.logpdf((x - self.params.alpha - self.params.beta * p)/abs(self.params.gamma))
            return np.log(1/abs(self.params.mu)) + norm.logpdf((x - p)/abs(self.params.mu))

        res = self.integrator_cls(**(self.integrator_params or {})).compute(integrand)
        return res.value, res.error
