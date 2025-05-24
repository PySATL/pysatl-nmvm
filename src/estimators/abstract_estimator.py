from typing import Self

from numpy import _typing

from src.procedures import ALGORITHM_REGISTRY
from src.estimators.estimate_result import EstimateResult
from src.register.algorithm_purpose import AlgorithmPurpose


class AbstractEstimator:
    """Base class for Estimators

    Attributes:
        algorithm_name: A string indicating chosen algorithm.
        params: A dictionary of algorithm parameters.
        estimate_result: Estimation result.
        _registry: Registry that contains classes of all algorithms.
        _purpose: Defines purpose of algorithm, one of the registry key.
    """

    def __init__(self, algorithm_name: str, params: dict | None = None) -> None:
        """Initializes the instance based on algorithm name and params.

        Args:
            algorithm_name: A string indicating chosen algorithm.
            params: A dictionary of algorithm parameters.
        """

        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        self.estimate_result = EstimateResult()
        self._registry = ALGORITHM_REGISTRY
        self._purpose = AlgorithmPurpose.DEFAULT

    def get_params(self) -> dict:
        return {"algorithm_name": self.algorithm_name, "params": self.params, "estimated_result": self.estimate_result}

    def set_params(self, algorithm_name: str, params: dict | None = None) -> Self:
        self.algorithm_name = algorithm_name
        if params is None:
            self.params = dict()
        else:
            self.params = params
        return self

    def get_available_algorithms(self) -> list[str]:
        """Get all algorithms that can be used for current estimator class"""
        return [key[0] for key in self._registry.register_of_names.keys() if key[1] == self._purpose]

    def estimate(self, sample: _typing.ArrayLike) -> EstimateResult:
        """Applies an algorithm to the given sample

        Args:
            sample: sample of the analysed distribution

        """
        cls = None
        if (self.algorithm_name, self._purpose) in self._registry.register_of_names:
            cls = self._registry.dispatch(self.algorithm_name, self._purpose)(sample, **self.params)
        if cls is None:
            raise ValueError("This algorithm does not exist")
        self.estimate_result = cls.algorithm(sample)
        return self.estimate_result
