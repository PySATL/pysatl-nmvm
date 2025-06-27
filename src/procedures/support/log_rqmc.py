from typing import Callable
import numpy as np
import numpy._typing as tpg
import scipy
from src.procedures.support.integrator import IntegrationResult
from src.procedures.support.rqmc import RQMC


class LogRQMCIntegrator:
    """Log-sum-exp stabilized randomized Quasi-Monte Carlo integrator."""

    def __init__(
        self,
        error_tolerance: float = 1e-6,
        count: int = 25,
        base_n: int = 2 ** 6,
        i_max: int = 100,
        a: float = 0.00047,
    ):
        self.error_tolerance = error_tolerance
        self.count = count
        self.base_n = base_n
        self.i_max = i_max
        self.a = a

    @staticmethod
    def lse(args: tpg.NDArray) -> float:
        """
        Log-Sum-Exp stabilization helper.
        """
        max_value = max(args)
        return max_value + np.log(np.sum(np.exp(args - max_value)))

    def _independent_estimator(self, values: tpg.NDArray) -> float:
        vfunc = np.vectorize(self.func)
        return -np.log(len(values)) + self.lse(vfunc(values))

    def _estimator(self, random_matrix: tpg.NDArray) -> tuple[float, tpg.NDArray]:
        values = np.array(list(map(self._independent_estimator, random_matrix)))
        return -np.log(self.count) + self.lse(values), values

    def _update_independent_estimator(
        self, i: int, old_value: float, new_values: tpg.NDArray
    ) -> float:
        return -np.log(i + 1) + self.lse(
            np.array(i * [old_value] + [self._independent_estimator(new_values[i * self.base_n :])])
        )

    def _update(
        self, j: int, old_values: tpg.NDArray, random_matrix: tpg.NDArray
    ) -> tuple[float, tpg.NDArray]:
        values = []
        for idx in range(self.count):
            old_val, new_vals = old_values[idx], random_matrix[idx]
            values.append(self._update_independent_estimator(j, old_val, new_vals))
        np_values = np.array(values)
        return -np.log(self.count) + self.lse(np_values), np_values

    def compute(self, func: Callable) -> IntegrationResult:
        """Compute integral of `func` over [0,1] with log-RQMC."""
        # Assign the integrand
        self.func = func
        # Instantiate underlying RQMC with same parameters
        rqmc_inst = RQMC(
            func,
            error_tolerance=self.error_tolerance,
            count=self.count,
            base_n=self.base_n,
            i_max=self.i_max,
            a=self.a,
        )
        # __call__ returns (approximation, error)
        approximation, error = rqmc_inst()
        return IntegrationResult(approximation, error, None)


if __name__ == "__main__":
    # Sample usage
    integrator = LogRQMCIntegrator(i_max=100)
    result = integrator.compute(lambda x: x**3 - x**2 + 1)
    print(result)
