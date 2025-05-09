from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.log_rqmc import LogRQMC
from src.algorithms.support_algorithms.rqmc import RQMC
from src.mixtures.abstract_mixture import AbstractMixtures


@dataclass
class _NMMClassicDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of classical NMM"""
    alpha: float | int | np.int64
    beta: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMMCanonicalDataCollector:
    """TODO: Change typing from float | int | etc to Protocol with __addition__ __multiplication__ __subtraction__"""

    """Data Collector for parameters of canonical NMM"""
    sigma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


class NormalMeanMixtures(AbstractMixtures):
    _classical_collector = _NMMClassicDataCollector
    _canonical_collector = _NMMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        """
        Read Doc of Parent Method
        """

        super().__init__(mixture_form, **kwargs)

    def _params_validation(self, data_collector: Any, params: dict[str, float | rv_continuous | rv_frozen]) -> Any:
        """
        Read parent method doc

        Raises:
            ValueError: If canonical Mixture has negative sigma parameter

        """

        data_class = super()._params_validation(data_collector, params)
        if hasattr(data_class, "sigma") and data_class.sigma <= 0:
            raise ValueError("Sigma cant be zero or negative")
        if hasattr(data_class, "gamma") and data_class.gamma == 0:
            raise ValueError("Gamma cant be zero")
        return data_class

    def _classical_moment(self, n: int, params: dict) -> tuple[float, float]:
        """
        Compute n-th moment of classical NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm

        Returns: moment approximation and error tolerance

        """
        mixture_moment = 0
        error_tolerance = 0
        for k in range(0, n + 1):
            for l in range(0, k + 1):
                coefficient = binom(n, n - k) * binom(k, k - l) * (self.params.beta ** (k - l)) * (self.params.gamma**l)
                mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (k - l), 0, 1, **params)
                error_tolerance += (self.params.beta ** (k - l)) * mixing_moment[1]
                mixture_moment += coefficient * (self.params.alpha ** (n - k)) * mixing_moment[0] * norm.moment(l)
        return mixture_moment, error_tolerance

    def _canonical_moment(self, n: int, params: dict) -> tuple[float, float]:
        """
        Compute n-th moment of canonical NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm

        Returns: moment approximation and error tolerance

        """
        mixture_moment = 0
        error_tolerance = 0
        for k in range(0, n + 1):
            coefficient = binom(n, n - k) * (self.params.sigma**k)
            mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (n - k), 0, 1, **params)
            error_tolerance += mixing_moment[1]
            mixture_moment += coefficient * mixing_moment[0] * norm.moment(k)
        return mixture_moment, error_tolerance

    def compute_moment(self, n: int, params: dict) -> tuple[float, float]:
        """
        Compute n-th moment of  NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm

        Returns: moment approximation and error tolerance

        """
        if isinstance(self.params, _NMMClassicDataCollector):
            return self._classical_moment(n, params)
        return self._canonical_moment(n, params)

    def _canonical_compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for canonical cdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed cdf and error tolerance

        """
        rqmc = RQMC(lambda u: norm.cdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)), **params)
        return rqmc()

    def _classical_compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for classic cdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed cdf and error tolerance

        """
        rqmc = RQMC(
            lambda u: norm.cdf(
                (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)
            ),
            **params
        )
        return rqmc()

    def compute_cdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Choose equation for cdf estimation depends on Mixture form
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: Computed pdf and error tolerance

        """
        if isinstance(self.params, _NMMCanonicalDataCollector):
            return self._canonical_compute_cdf(x, params)
        return self._classical_compute_cdf(x, params)

    def _canonical_compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for canonical pdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed pdf and error tolerance

        """
        rqmc = RQMC(
            lambda u: (1 / np.abs(self.params.sigma))
            * norm.pdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
            **params
        )
        return rqmc()

    def _classical_compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for classic pdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed pdf and error tolerance

        """
        rqmc = RQMC(
            lambda u: (1 / np.abs(self.params.gamma))
            * norm.pdf(
                (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)
            ),
            **params
        )
        return rqmc()

    def compute_pdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Choose equation for pdf estimation depends on Mixture form
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: Computed pdf and error tolerance

        """
        if isinstance(self.params, _NMMCanonicalDataCollector):
            return self._canonical_compute_pdf(x, params)
        return self._classical_compute_pdf(x, params)

    def _classical_compute_log_pdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for classic log pdf
        Args:
            x (): point
            params (): parameters of LogRQMC algorithm

        Returns: computed log pdf and error tolerance

        """
        rqmc = LogRQMC(
            lambda u: np.log(1 / np.abs(self.params.gamma))
            + norm.logpdf(
                (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)
            ),
            **params
        )
        return rqmc()

    def _canonical_compute_log_pdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Equation for canonical log pdf
        Args:
            x (): point
            params (): parameters of LogRQMC algorithm

        Returns: computed log pdf and error tolerance

        """
        rqmc = LogRQMC(
            lambda u: np.log(1 / np.abs(self.params.sigma))
            + norm.logpdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
            **params
        )
        return rqmc()

    def compute_logpdf(self, x: float, params: dict) -> tuple[float, float]:
        """
        Choose equation for log pdf estimation depends on Mixture form
        Args:
            x (): point
            params (): parameters of LogRQMC algorithm

        Returns: Computed log pdf and error tolerance

        """
        if isinstance(self.params, _NMMCanonicalDataCollector):
            return self._canonical_compute_log_pdf(x, params)
        return self._classical_compute_log_pdf(x, params)
