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
class _NVMClassicDataCollector:
    alpha: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous

@dataclass
class _NVMCanonicalDataCollector:
    alpha: float | int | np.int64
    distribution: rv_frozen | rv_continuous

class NormalVarianceMixtures(AbstractMixtures):
    _classical_collector = _NVMClassicDataCollector
    _canonical_collector = _NVMCanonicalDataCollector

    def __init__(
        self,
        mixture_form: str,
        integrator_cls: Type[Integrator] = RQMCIntegrator,
        integrator_params: Dict[str, Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(mixture_form, **kwargs)
        self.integrator_cls = integrator_cls
        self.integrator_params = integrator_params or {}

    def _compute_moment(self, n: int, __: Dict[str, Any]) -> tuple[float, float]:
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
        integrator = self.integrator_cls(**self.integrator_params)
        result = integrator.compute(integrand)
        return result.value, result.error

    def _compute_cdf(self, x: float, __: Dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        param_norm = norm(0, gamma)
        def integrand(u: float) -> float:
            return param_norm.cdf((x - self.params.alpha) / np.sqrt(self.params.distribution.ppf(u)))
        integrator = self.integrator_cls(**self.integrator_params)
        result = integrator.compute(integrand)
        return result.value, result.error

    def _compute_pdf(self, x: float, __: Dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        d = (x - self.params.alpha) ** 2 / gamma ** 2
        def integrand(u: float) -> float:
            return self._integrand_func(u, d, gamma)
        integrator = self.integrator_cls(**self.integrator_params)
        result = integrator.compute(integrand)
        return result.value, result.error

    def _compute_logpdf(self, x: float, __: Dict[str, Any]) -> tuple[float, float]:
        gamma = getattr(self.params, 'gamma', 1)
        d = (x - self.params.alpha) ** 2 / gamma ** 2
        def integrand(u: float) -> float:
            return self._log_integrand_func(u, d, gamma)
        # For log-pdf you may choose LogRQMC or any integrator that supports it
        integrator = self.integrator_cls(**self.integrator_params)
        result = integrator.compute(integrand)
        return result.value, result.error

    @lru_cache()
    def _integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return (1 / np.sqrt(2 * np.pi * ppf * np.abs(gamma ** 2))) * np.exp(-d / (2 * ppf))

    def _log_integrand_func(self, u: float, d: float, gamma: float) -> float:
        ppf = self.params.distribution.ppf(u)
        return -(ppf * np.log(2 * np.pi * ppf * gamma ** 2) + d) / (2 * ppf)
