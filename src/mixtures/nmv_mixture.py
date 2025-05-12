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
        super().__init__(mixture_form, **kwargs)

    def _classical_moment(self, n: int, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        """
        Compute n-th moment of classical NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm
            integrator (): type of integrator to computing

        Returns: moment approximation and error tolerance

        """

        def integral_func(u: float) -> float:
            result = 0
            for k in range(0, n + 1):
                for l in range(0, k + 1):
                    result += (
                        binom(n, n - k)
                        * binom(k, k - l)
                        * (self.params.beta ** (k - l))
                        * (self.params.gamma**l)
                        * self.params.distribution.ppf(u) ** (k - l / 2)
                        * (self.params.alpha ** (n - k))
                        * norm.moment(l)
                    )
            return result

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: integral_func(u), **params)
        return rqmc.value, rqmc.error

    def _canonical_moment(self, n: int, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        """
        Compute n-th moment of classical NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm
            integrator (): type of integrator to computing

        Returns: moment approximation and error tolerance

        """

        def integral_func(u: float) -> float:
            result = 0
            for k in range(0, n + 1):
                for l in range(0, k + 1):
                    result += (
                        binom(n, n - k)
                        * binom(k, k - l)
                        * (self.params.nu ** (k - l))
                        * self.params.distribution.ppf(u) ** (k - l / 2)
                        * (self.params.alpha ** (n - k))
                        * norm.moment(l)
                    )
            return result

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: integral_func(u), **params)
        return rqmc.value, rqmc.error

    def compute_moment(self, n: int, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        if isinstance(self.params, _NMVMClassicDataCollector):
            return self._classical_moment(n, params, integrator)
        return self._canonical_moment(n, params, integrator)

    def _classical_cdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = lru_cache()(self.params.distribution.ppf)(u)
            point = (x - self.params.alpha) / (np.sqrt(ppf) * self.params.gamma) - (
                self.params.beta / self.params.gamma * np.sqrt(ppf)
            )
            return norm.cdf(point)

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: _inner_func(u), **params)
        return rqmc.value, rqmc.error

    def _canonical_cdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            point = (x - self.params.alpha) / (np.sqrt(ppf)) - (self.params.mu * np.sqrt(ppf))
            return norm.cdf(point)

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: _inner_func(u), **params)
        return rqmc.value, rqmc.error

    def compute_cdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        if isinstance(self.params, _NMVMClassicDataCollector):
            return self._classical_cdf(x, params, integrator)
        return self._canonical_cdf(x, params, integrator)

    def _classical_pdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            return (
                1
                / np.sqrt(2 * np.pi * ppf * self.params.gamma**2)
                * np.exp(
                    -((x - self.params.alpha) ** 2 + self.params.beta**2 * ppf**2) / (2 * ppf * self.params.gamma**2)
                )
            )

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: _inner_func(u), **params)
        return np.exp(self.params.beta * (x - self.params.alpha) / self.params.gamma**2) * rqmc.value, rqmc.error

    def _canonical_pdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            return (
                1
                / np.sqrt(2 * np.pi * ppf)
                * np.exp(-((x - self.params.alpha) ** 2 + self.params.mu**2 * ppf**2) / (2 * ppf))
            )

        integrator = integrator or RQMCIntegrator()
        rqmc = integrator.compute_integral(func=lambda u: _inner_func(u), **params)
        return np.exp(self.params.mu * (x - self.params.alpha)) * rqmc.value, rqmc.error

    def _classical_log_pdf(self, x: float, params: dict) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            return -(
                (x - self.params.alpha) ** 2
                + ppf**2 * self.params.beta**2
                + ppf * self.params.gamma**2 * np.log(2 * np.pi * ppf * self.params.gamma**2)
            ) / (2 * ppf * self.params.gamma**2)

        rqmc = LogRQMC(lambda u: _inner_func(u), **params)
        return rqmc()

    def _canonical_log_pdf(self, x: float, params: dict) -> tuple[float, float]:
        def _inner_func(u: float) -> float:
            ppf = self.params.distribution.ppf(u)
            return -((x - self.params.alpha) ** 2 + ppf**2 * self.params.mu**2 + ppf * np.log(2 * np.pi * ppf)) / (
                2 * ppf
            )

        rqmc = LogRQMC(lambda u: _inner_func(u), **params)
        return rqmc()

    def compute_pdf(self, x: float, params: dict, integrator: Integrator = None) -> tuple[float, float]:
        if isinstance(self.params, _NMVMClassicDataCollector):
            return self._classical_pdf(x, params, integrator)
        return self._canonical_pdf(x, params, integrator)

    def compute_logpdf(self, x: float, params: dict) -> tuple[Any, float]:
        if isinstance(self.params, _NMVMClassicDataCollector):
            int_result = self._classical_log_pdf(x, params)
            return self.params.beta * (x - self.params.alpha) / self.params.gamma**2 + int_result[0], int_result[1]
        int_result = self._canonical_log_pdf(x, params)
        return self.params.mu * (x - self.params.alpha) + int_result[0], int_result[1]
