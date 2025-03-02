from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.rqmc import RQMC
from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NMVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NMVM"""
    alpha: float | int
    beta: float | int
    gamma: float | int
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NMVM"""
    alpha: float | int
    mu: float | int
    distribution: rv_frozen | rv_continuous


class NormalMeanVarianceMixtures(AbstractMixtures):
    _classical_collector = _NMVMClassicDataCollector
    _canonical_collector = _NMVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def compute_moment(self, n: int, params: dict) -> tuple[float, float]:
        pass

    def _classical_cdf(self, s: float, x: float) -> float:
        beta = self.params.beta
        gamma = self.params.gamma if isinstance(self.params, _NMVMClassicDataCollector) else 1
        parametric_norm = norm(0, gamma)
        return parametric_norm.cdf(
            (x - self.params.alpha) / np.sqrt(self.params.distribution.ppf(s))
            - beta / gamma**2 * np.sqrt(self.params.distribution.ppf(s))
        )

    def compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        rqmc = RQMC(lambda s: self._classical_cdf(s, x), **params)
        return rqmc()

    def _cdf_under_func(self, s: float, x: float) -> float:
        def first_exp(s: float) -> float:
            return np.exp(-((x**2) / 2) / s)

        def second_exp(s: float, beta: float) -> float:
            return np.exp(beta**2 * s / 2)

        def sqrt_part(s: float) -> float:
            return np.sqrt(2 * np.pi * self.params.distribution.ppf(s))

        if isinstance(self.params, _NMVMClassicDataCollector):
            return first_exp(s) * second_exp(s, self.params.beta) / sqrt_part(s)
        return first_exp(s) * second_exp(s, self.params.mu) / sqrt_part(s)

    def compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        rqmc = RQMC(lambda s: self._cdf_under_func(s, x), **params)
        return np.exp(self.params.mu * x) * rqmc()

    def compute_logpdf(self, x: float, params: dict) -> tuple[float, float]:
        log_rqmc = LogRQMC(lambda s: self._cdf_under_func(s, x), **params)
        return self.params.mu * x * log_rqmc()
