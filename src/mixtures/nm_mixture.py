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

    def compute_moment(self, values: list, params: dict) -> list[set[int | Any]]:
        """
        Compute n-th moment of classical/canonical NMM

        Args:
            n (): Moment ordinal
            params (): Parameters of integration algorithm

        Returns: moment approximation and error tolerance

        """
        mixture_moment = 0
        error_tolerance = 0
        ans=[]
        if self.mixture_form == "classical":
            for n in values:
                for k in range(0, n + 1):
                    for l in range(0, k + 1):
                        coefficient = binom(n, n - k) * binom(k, k - l) * (self.params.beta ** (k - l)) * (self.params.gamma**l)
                        mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (k - l), 0, 1, **params)
                        error_tolerance += (self.params.beta ** (k - l)) * mixing_moment[1]
                        mixture_moment += coefficient * (self.params.alpha ** (n - k)) * mixing_moment[0] * norm.moment(l)
                ans.append({mixture_moment, error_tolerance})
        for n in values:
            for k in range(0, n + 1):
                coefficient = binom(n, n - k) * (self.params.sigma**k)
                mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (n - k), 0, 1, **params)
                error_tolerance += mixing_moment[1]
                mixture_moment += coefficient * mixing_moment[0] * norm.moment(k)
            ans.append({mixture_moment, error_tolerance})
        return ans



    def compute_cdf(self, values: list, params: dict) -> list[tuple[float, float]]:
        """
        Equation for classical/canonical cdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed cdf and error tolerance

        """
        if self.mixture_form=="classical":
            rqmcs = [
                RQMC(
                lambda u: norm.cdf(
                    (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)
                ),
                **params
            )
            for x in values
            ]
        else:
            rqmcs = [RQMC(lambda u: norm.cdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)), **params) for x in values]

        return [rqmc() for rqmc in rqmcs]



    def compute_pdf(self, values: list, params: dict) -> list[tuple[float, float]]:
        """
        Equation for classical/canonical pdf
        Args:
            x (): point
            params (): parameters of RQMC algorithm

        Returns: computed pdf and error tolerance

        """
        if self.mixture_form=="classical":
            rqmcs = [
                RQMC(
                lambda u: (1 / np.abs(self.params.gamma))
                * norm.pdf(
                    (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)
                ),
                **params
            )
            for x in values
            ]
        else:
            rqmcs = [
                RQMC(
                lambda u: (1 / np.abs(self.params.sigma))
                          * norm.pdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
                **params
            )
            for x in values
            ]

        return [rqmc() for rqmc in rqmcs]


    def compute_log_pdf(self, values: list, params: dict) -> list[tuple[float, float]]:
        """
        Equation for classical/canonical log pdf
        Args:
            x (): point
            params (): parameters of LogRQMC algorithm

        Returns: computed log pdf and error tolerance

        """
        if self.mixture_form=="classical":
            rqmcs = [
                LogRQMC(
                    lambda u, x=x: (
                            np.log(1 / np.abs(self.params.gamma)) +
                            norm.logpdf(
                                (x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(
                                    self.params.gamma)
                            )
                    ),
                    **params
                )
                for x in values
            ]
        else:
            rqmcs = [
                LogRQMC(
                    lambda u: np.log(1 / np.abs(self.params.sigma))
                          + norm.logpdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
                    **params
                )
                for x in values
            ]

        results = [rqmc() for rqmc in rqmcs]
        return results
