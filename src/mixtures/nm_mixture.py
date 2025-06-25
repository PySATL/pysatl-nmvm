from dataclasses import dataclass
from typing import Any, Type, Dict, Tuple

import numpy as np
from scipy.special import binom
from scipy.stats import norm, rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.integrator import Integrator
from src.mixtures.abstract_mixture import AbstractMixtures

@dataclass
class _NMMClassicDataCollector:
    alpha: float | int | np.int64
    beta:  float | int | np.int64
    gamma: float | int | np.int64
    distribution: rv_frozen | rv_continuous

@dataclass
class _NMMCanonicalDataCollector:
    sigma: float | int | np.int64
    distribution: rv_frozen | rv_continuous

class NormalMeanMixtures(AbstractMixtures):
    _classical_collector = _NMMClassicDataCollector
    _canonical_collector = _NMMCanonicalDataCollector

    def __init__(
        self,
        mixture_form: str,
        integrator_cls: Type[Integrator] = Integrator,
        integrator_params: Dict[str, Any] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(mixture_form, integrator_cls=integrator_cls, integrator_params=integrator_params, **kwargs)

    def _params_validation(self, data_collector: Any, params: dict[str, float | rv_continuous | rv_frozen]) -> Any:
        data_class = super()._params_validation(data_collector, params)
        if hasattr(data_class, "sigma") and data_class.sigma <= 0:
            raise ValueError("Sigma can't be zero or negative")
        if hasattr(data_class, "gamma") and data_class.gamma == 0:
            raise ValueError("Gamma can't be zero")
        return data_class

    def _compute_moment(self, n: int) -> Tuple[float, float]:
        mixture_moment = 0.0
        error = 0.0
        if self.mixture_form == "classical":
            for k in range(n + 1):
                for l in range(k + 1):
                    coeff = binom(n, n - k) * binom(k, k - l)
                    def mix(u: float) -> float:
                        return (
                            self.params.distribution.ppf(u) ** (k - l)
                        )
                    res = self.integrator_cls(**(self.integrator_params or {})).compute(mix)
                    mixture_moment += coeff * (self.params.beta ** (k - l)) * (self.params.gamma ** l) * (self.params.alpha ** (n - k)) * res.value * norm.moment(l)
                    error += coeff * (self.params.beta ** (k - l)) * (self.params.gamma ** l) * (self.params.alpha ** (n - k)) * res.error * norm.moment(l)
        else:
            for k in range(n + 1):
                coeff = binom(n, n - k)
                def mix(u: float) -> float:
                    return self.params.distribution.ppf(u) ** (n - k)
                res = self.integrator_cls(**(self.integrator_params or {})).compute(mix)
                mixture_moment += coeff * (self.params.sigma ** k) * res.value * norm.moment(k)
                error += coeff * (self.params.sigma ** k) * res.error * norm.moment(k)
        return mixture_moment, error

    def _compute_cdf(self, x: float) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return norm.cdf((x - self.params.alpha - self.params.beta * p) / abs(self.params.gamma))
        else:
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return norm.cdf((x - p) / abs(self.params.sigma))
        res = self.integrator_cls(**(self.integrator_params or {})).compute(fn)
        return res.value, res.error

    def _compute_pdf(self, x: float) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return (1 / abs(self.params.gamma)) * norm.pdf((x - self.params.alpha - self.params.beta * p) / abs(self.params.gamma))
        else:
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return (1 / abs(self.params.sigma)) * norm.pdf((x - p) / abs(self.params.sigma))
        res = self.integrator_cls(**(self.integrator_params or {})).compute(fn)
        return res.value, res.error

    def _compute_logpdf(self, x: float) -> Tuple[float, float]:
        if self.mixture_form == "classical":
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return np.log(1 / abs(self.params.gamma)) + norm.logpdf((x - self.params.alpha - self.params.beta * p) / abs(self.params.gamma))
        else:
            def fn(u: float) -> float:
                p = self.params.distribution.ppf(u)
                return np.log(1 / abs(self.params.sigma)) + norm.logpdf((x - p) / abs(self.params.sigma))
        res = self.integrator_cls(**(self.integrator_params or {})).compute(fn)
        return res.value, res.error
