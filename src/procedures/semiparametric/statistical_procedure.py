from typing import Protocol
import numpy as np

from src.estimators.estimate_result import EstimateResult

class StatisticalProcedure(Protocol):
    """Protocol for statistical procedures that compute estimates from data.

    """
    def compute(self, sample: np.ndarray) -> EstimateResult:
        pass

