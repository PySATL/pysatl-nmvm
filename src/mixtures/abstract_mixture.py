from abc import ABCMeta, abstractmethod
from dataclasses import fields
from typing import Any, List, Tuple, Union, Dict
import numpy as np
from numpy.typing import NDArray

from scipy.stats import rv_continuous
from scipy.stats.distributions import rv_frozen


class AbstractMixtures(metaclass=ABCMeta):
    """Base class for Mixtures"""

    _classical_collector: Any
    _canonical_collector: Any

    @abstractmethod
    def __init__(self, mixture_form: str, **kwargs: Any) -> None:
        """
        Args:
            mixture_form: Form of Mixture classical or Canonical
            **kwargs: Parameters of Mixture
        """
        if mixture_form == "classical":
            self.params = self._params_validation(self._classical_collector, kwargs)
        elif mixture_form == "canonical":
            self.params = self._params_validation(self._canonical_collector, kwargs)
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")

    @abstractmethod
    def _compute_moment(self, n: int, params: Dict) -> Tuple[float, float]:
        ...

    def compute_moment(
        self, x: Union[List[int], int, NDArray[np.float64]], params: Dict
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_moment(xp, params) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_moment(xp, params) for xp in x]
        elif isinstance(x, int):
            return self._compute_moment(x, params)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_pdf(self, x: float, params: Dict) -> Tuple[float, float]:
        ...

    def compute_pdf(
        self, x: Union[List[float], float, NDArray[np.float64]], params: Dict
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_pdf(xp, params) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_pdf(xp, params) for xp in x]
        elif isinstance(x, float):
            return self._compute_pdf(x, params)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_logpdf(self, x: float, params: Dict) -> Tuple[float, float]:
        ...

    def compute_logpdf(
        self, x: Union[List[float], float, NDArray[np.float64]], params: Dict
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_logpdf(xp, params) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_logpdf(xp, params) for xp in x]
        elif isinstance(x, float):
            return self._compute_logpdf(x, params)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    @abstractmethod
    def _compute_cdf(self, x: float, rqmc_params: Dict[str, Any]) -> Tuple[float, float]:
        ...

    def compute_cdf(
        self, x: Union[List[float], float, NDArray[np.float64]], params: Dict
    ) -> Union[List[Tuple[float, float]], Tuple[float, float], NDArray[Any]]:
        if isinstance(x, np.ndarray):
            return np.array([self._compute_cdf(xp, params) for xp in x], dtype=object)
        elif isinstance(x, list):
            return [self._compute_cdf(xp, params) for xp in x]
        elif isinstance(x, float):
            return self._compute_cdf(x, params)
        else:
            raise TypeError(f"Unsupported type for x: {type(x)}")

    def _params_validation(self, data_collector: Any, params: dict[str, float | rv_continuous | rv_frozen]) -> Any:
        """Mixture Parameters Validation

        Args:
            data_collector: Dataclass that collect parameters of Mixture
            params: Input parameters

        Returns: Instance of dataclass

        Raises:
            ValueError: If given parameters is unexpected
            ValueError: If parameter type is invalid
            ValueError: If parameters age not given

        """

        dataclass_fields = fields(data_collector)
        if len(params) != len(dataclass_fields):
            raise ValueError(f"Expected {len(dataclass_fields)} arguments, got {len(params)}")
        names_and_types = dict((field.name, field.type) for field in dataclass_fields)
        for pair in params.items():
            if pair[0] not in names_and_types:
                raise ValueError(f"Unexpected parameter {pair[0]}")
            if not isinstance(pair[1], names_and_types[pair[0]]):
                raise ValueError(f"Type missmatch: {pair[0]} should be {names_and_types[pair[0]]}, not {type(pair[1])}")
        return data_collector(**params)