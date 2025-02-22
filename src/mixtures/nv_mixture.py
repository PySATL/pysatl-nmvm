from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.rqmc import RQMC
from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NVMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NVM"""
    alpha: float | int
    gamma: float | int
    distribution: rv_frozen | rv_continuous


@dataclass
class _NVMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NVM"""
    alpha: float | int
    distribution: rv_frozen | rv_continuous


class NormalVarianceMixtures(AbstractMixtures):

    _classical_collector = _NVMClassicDataCollector
    _canonical_collector = _NVMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def compute_moment(self, n: int, params: dict) -> tuple[float, float]:
        raise NotImplementedError("Must implement compute_moment")

    def compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        rqmc = RQMC(
            lambda u: norm(0, self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1).cdf(
                x / np.sqrt(self.params.distribution.ppf(u))
            ),
            **params
        )
        return rqmc()

    def _integrand_func(self, u: float, d: float) -> float:
        gamma = self.params.gamma if isinstance(self.params, _NVMClassicDataCollector) else 1
        return (1 / (np.pi * 2 * self.params.distribution.ppf(u) * np.abs(gamma**2))) * np.exp(
            -1 * d / (2 * self.params.distribution.ppf(u))
        )

    def compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        d = (x - self.params.alpha) ** 2 / self.params.gamma**2
        rqmc = RQMC(lambda u: self._integrand_func(u, d), **params)
        return rqmc()

    def compute_logpdf(self, x: float, params: dict) -> tuple[float, float]:
        d = (x - self.params.alpha) ** 2 / self.params.gamma**2
        log_rqmc = LogRQMC(lambda u: self._integrand_func(u, d), **params)
        return log_rqmc()
