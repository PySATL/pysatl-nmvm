from typing import Callable

import numpy as np
import numpy._typing as tpg
import scipy

from src.algorithms.support_algorithms.rqmc import RQMC


class LogRQMC(RQMC):
    def __init__(
        self,
        func: Callable,
        error_tolerance: float = 1e-6,
        count: int = 25,
        base_n: int = 2**6,
        i_max: int = 100,
        a: float = 0.00047,
    ):
        super().__init__(func, error_tolerance, count, base_n, i_max, a)

    @staticmethod
    def lse(args: tpg.NDArray) -> float:
        """
        Compute LSE
        Args:
            args (): Values

        Returns: LSE result

        """
        max_value = max(args)
        return max_value + np.log(np.sum(np.exp(args - max_value)))

    def _independent_estimator(self, values: np._typing.NDArray) -> float:
        """Apply function to row of matrix and find mean of row

        Args:
            values: row of random values matrix

        Returns: mean of row

        """
        vfunc = np.vectorize(self.func)
        return -np.log(len(values)) + self.lse(vfunc(values))

    def _estimator(self, random_matrix: np._typing.NDArray) -> tuple[float, np._typing.NDArray]:
        """Find mean of all rows

        Args:
            random_matrix: matrix of random values

        Returns: mean of all rows

        """
        values = np.array(list(map(self._independent_estimator, random_matrix)))
        return -np.log(self.count) + self.lse(values), values

    def _update_independent_estimator(self, i: int, old_value: float, new_values: np._typing.NDArray) -> float:
        """Update mean of row

        Args:
            i: step count
            old_value: previous value of row on i-1 step
            new_values: new generated row of random values

        Returns: Updated mean of row

        """

        return -np.log(i + 1) + self.lse(
            np.array(i * [old_value] + [self._independent_estimator(new_values[i * self.base_n :])])
        )

    def _update(
        self, j: int, old_values: np._typing.NDArray, random_matrix: np._typing.NDArray
    ) -> tuple[float, np._typing.NDArray]:
        """Update mean of all rows

        Args:
            j: step count
            old_values: previous values of row on i-1 step
            random_matrix: new generated matrix of random values

        Returns:Updated mean of all rows

        """
        values = []
        for i in range(self.count):
            old_value, new_values = old_values[i], random_matrix[i]
            value = self._update_independent_estimator(j, old_value, new_values)
            values.append(value)
        np_values = np.array(values)
        return -np.log(self.count) + self.lse(np_values), np_values

    def rqmc(self) -> tuple[float, float]:
        """Main function of algorithm

        Returns: approximation for integral of function from 0 to 1

        """
        sobol_sampler = scipy.stats.qmc.Sobol(d=1, scramble=False)
        sobol_sample = np.repeat(sobol_sampler.random(self.base_n).transpose(), self.count, axis=0)
        xor_sample = np.array(np.random.rand(1, self.count)[0])
        sample = self._digital_shift(sobol_sample, xor_sample)
        approximation, values = self._estimator(sample)
        current_error_tolerance = self._sigma(values, approximation) * self.z
        for i in range(1, self.i_max):
            if current_error_tolerance < self.error_tolerance:
                return approximation, current_error_tolerance
            sobol_sampler.reset()
            sobol_sample = np.repeat(sobol_sampler.random(self.base_n * (i + 1)).transpose(), self.count, axis=0)
            sample = self._digital_shift(sobol_sample, xor_sample)
            approximation, values = self._update(i, values, sample)
            current_error_tolerance = self._sigma(values, approximation) * self.z
        return approximation, current_error_tolerance


if __name__ == "__main__":
    log_rqmc = LogRQMC(lambda x: x**3 - x**2 + 1, i_max=100)
    print(log_rqmc())
