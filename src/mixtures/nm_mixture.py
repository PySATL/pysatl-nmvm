from dataclasses import dataclass
from typing import Any, List, Tuple

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
    alpha: float | int | np.int64
    beta: float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


@dataclass
class _NMMCanonicalDataCollector:
    sigma: float | int | np.int64
    distribution: rv_frozen | rv_continuous


class NormalMeanMixtures(AbstractMixtures):
    _classical_collector = _NMMClassicDataCollector
    _canonical_collector = _NMMCanonicalDataCollector

    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        super().__init__(mixture_form, **kwargs)

    def _params_validation(self, data_collector: Any, params: dict[str, float | rv_continuous | rv_frozen]) -> Any:
        data_class = super()._params_validation(data_collector, params)
        if hasattr(data_class, "sigma") and data_class.sigma <= 0:
            raise ValueError("Sigma can't be zero or negative")
        if hasattr(data_class, "gamma") and data_class.gamma == 0:
            raise ValueError("Gamma can't be zero")
        return data_class

    def compute_moment(self, n: int, params: dict) -> Tuple[float, float]:
        mixture_moment = 0
        error_tolerance = 0
        if self.mixture_form == "classical":
            for k in range(0, n + 1):
                for l in range(0, k + 1):
                    coefficient = binom(n, n - k) * binom(k, k - l) * (self.params.beta ** (k - l)) * (self.params.gamma ** l)
                    mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (k - l), 0, 1, **params)
                    error_tolerance += (self.params.beta ** (k - l)) * mixing_moment[1]
                    mixture_moment += coefficient * (self.params.alpha ** (n - k)) * mixing_moment[0] * norm.moment(l)
        else:
            for k in range(0, n + 1):
                coefficient = binom(n, n - k) * (self.params.sigma ** k)
                mixing_moment = quad(lambda u: self.params.distribution.ppf(u) ** (n - k), 0, 1, **params)
                error_tolerance += mixing_moment[1]
                mixture_moment += coefficient * mixing_moment[0] * norm.moment(k)
        return mixture_moment, error_tolerance

    def compute_moments(self, values: List[int], params: dict) -> List[Tuple[float, float]]:
        return [self.compute_moment(n, params) for n in values]

    def compute_cdf(self, x: float, params: dict) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            rqmc = RQMC(
                lambda u: norm.cdf((x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)),
                **params,
            )
        else:
            rqmc = RQMC(
                lambda u: norm.cdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
                **params,
            )
        return rqmc()

    def compute_cdfs(self, values: List[float], params: dict) -> List[Tuple[float, float]]:
        return [self.compute_cdf(x, params) for x in values]

    def compute_pdf(self, x: float, params: dict) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            rqmc = RQMC(
                lambda u: (1 / np.abs(self.params.gamma))
                * norm.pdf((x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma)),
                **params,
            )
        else:
            rqmc = RQMC(
                lambda u: (1 / np.abs(self.params.sigma)) * norm.pdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
                **params,
            )
        return rqmc()

    def compute_pdfs(self, values: List[float], params: dict) -> List[Tuple[float, float]]:
        return [self.compute_pdf(x, params) for x in values]

    def compute_logpdf(self, x: float, params: dict) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            rqmc = LogRQMC(
                lambda u: (
                    np.log(1 / np.abs(self.params.gamma))
                    + norm.logpdf((x - self.params.alpha - self.params.beta * self.params.distribution.ppf(u)) / np.abs(self.params.gamma))
                ),
                **params,
            )
        else:
            rqmc = LogRQMC(
                lambda u: np.log(1 / np.abs(self.params.sigma)) + norm.logpdf((x - self.params.distribution.ppf(u)) / np.abs(self.params.sigma)),
                **params,
            )
        return rqmc()

    def compute_log_pdfs(self, values: List[float], params: dict) -> List[Tuple[float, float]]:
        return [self.compute_logpdf(x, params) for x in values]
