from abc import ABCMeta, abstractmethod
from dataclasses import fields
from typing import Any

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
        self.mixture_form = mixture_form
        if mixture_form == "classical":
            self.params = self._params_validation(self._classical_collector, kwargs)
        elif mixture_form == "canonical":
            self.params = self._params_validation(self._canonical_collector, kwargs)
        else:
            raise AssertionError(f"Unknown mixture form: {mixture_form}")

    @abstractmethod
    def compute_moment(self, n: int, params: dict) -> tuple[float, float]: ...

    @abstractmethod
    def compute_cdf(self, x: float, params: dict) -> tuple[float, float]: ...

    @abstractmethod
    def compute_pdf(self, x: float, params: dict) -> tuple[float, float]: ...

    @abstractmethod
    def compute_logpdf(self, x: float, params: dict) -> tuple[float, float]: ...

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
        return data_collector(**params)
