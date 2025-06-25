from abc import ABCMeta, abstractmethod
from dataclasses import fields
from typing import Any, List, Tuple, Union, Dict, Type
import numpy as np
from numpy.typing import NDArray

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen

from src.algorithms.support_algorithms.integrator import Integrator
from src.algorithms.support_algorithms.rqmc import RQMCIntegrator  # default integrator

class AbstractMixtures(metaclass=ABCMeta):
    """Base class for Mixtures"""

    _classical_collector: Any
    _canonical_collector: Any

    def __init__(
        self,
        mixture_form: str,
        integrator_cls: Type[Integrator] = RQMCIntegrator,
        integrator_params: Dict[str, Any] = None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            mixture_form: Form of Mixture classical or canonical
            integrator_cls: Class implementing Integrator protocol (default: RQMCIntegrator)
            integrator_params: Parameters for integrator constructor (default: {{}})
            **kwargs: Parameters of Mixture (alpha, gamma, etc.)
        """
        self.mixture_form = mixture_form
        self.integrator_cls = integrator_cls
        self.integrator_params = integrator_params or {}

        if mixture_form == "classical":
            self.params = self._params_validation(self._classical_collector, kwargs)
        elif mixture_form == "canonical":
            self.params = self._params_validation(self._canonical_collector, kwargs)
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")

    @abstractmethod
    def _compute_moment(self, n: int) -> Tuple[float, float]:
        ...

    def compute_moment(
        self,
        x: Union[List[int], int, NDArray[np.float64]]
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_moment(xp) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_moment(xp) for xp in x]
        elif isinstance(x, int):
            return self._compute_moment(x)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_pdf(self, x: float) -> Tuple[float, float]:
        ...

    def compute_pdf(
        self,
        x: Union[List[float], float, NDArray[np.float64]]
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_pdf(xp) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_pdf(xp) for xp in x]
        elif isinstance(x, float):
            return self._compute_pdf(x)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_logpdf(self, x: float) -> Tuple[float, float]:
        ...

    def compute_logpdf(
        self,
        x: Union[List[float], float, NDArray[np.float64]]
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_logpdf(xp) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_logpdf(xp) for xp in x]
        elif isinstance(x, float):
            return self._compute_logpdf(x)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_cdf(self, x: float) -> Tuple[float, float]:
        ...

    def compute_cdf(
        self,
        x: Union[List[float], float, NDArray[np.float64]]
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_cdf(xp) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_cdf(xp) for xp in x]
        elif isinstance(x, float):
            return self._compute_cdf(x)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    def _params_validation(
        self,
        data_collector: Any,
        params: dict[str, float | rv_continuous | rv_frozen]
    ) -> Any:
        """Mixture Parameters Validation"""
        dataclass_fields = fields(data_collector)
        if len(params) != len(dataclass_fields):
            raise ValueError(f"Expected {len(dataclass_fields)} arguments, got {len(params)}")
        names_and_types = {field.name: field.type for field in dataclass_fields}
        for name, value in params.items():
            if name not in names_and_types:
                raise ValueError(f"Unexpected parameter {name}")
            if not isinstance(value, names_and_types[name]):
                raise ValueError(
                    f"Type mismatch: {name} should be {names_and_types[name]}, not {type(value)}"
                )
        return data_collector(**params)
