from typing import Callable

import numpy as np
import numpy._typing as tpg
import scipy
from numba import njit

from src.algorithms.support_algorithms.integrator import IntegrationResult

BITS = 30
"""Number of bits in XOR. Should be less than 64"""
NUMBA_FAST_MATH = True
"""Flag for Numba fastmath. May be less accurate in some cases"""


class RQMC:
    """Randomize Quasi Monte Carlo Method

    Args:
        func: integrated function
        error_tolerance: pre-specified error tolerance
        count: number of rows of random values matrix
        base_n: number of columns of random values matrix
        i_max: allowed number of cycles
        a: parameter for quantile of normal distribution

    """

    def __init__(
        self,
        func: Callable,
        error_tolerance: float = 1e-6,
        count: int = 25,
        base_n: int = 2**6,
        i_max: int = 100,
        a: float = 0.00047,
    ):
        self._args_validation(error_tolerance, count, base_n, i_max, a)
        self.func = func
        self.error_tolerance = error_tolerance
        self.count = count
        self.base_n = base_n
        self.i_max = i_max
        self.z = scipy.stats.norm.ppf(1 - a / 2)

        if NUMBA_FAST_MATH:
            setattr(self, "_xor_float", njit(fastmath=True)(RQMC._xor_float))

    @staticmethod
    def _args_validation(error_tolerance: float, count: int, base_n: int, i_max: int, a: float) -> None:
        """Parse arguments

        Args:
        func: integrated function
        error_tolerance: pre-specified error tolerance
        count: number of rows of random values matrix
        base_n: number of columns of random values matrix
        i_max: allowed number of cycles
        a: parameter for quantile of normal distribution

        Returns: None

        Raises:
            ValueError: if any argument is not positive
            ValueError: if base n is not power of 2
        """

        if error_tolerance < 0:
            raise ValueError("Error tolerance must be positive")
        if count <= 0:
            raise ValueError("Count must be positive")
        if base_n <= 0:
            raise ValueError("Base n must be positive")
        if base_n & (base_n - 1) != 0:
            raise ValueError("Base n must be power of 2")
        if i_max <= 0:
            raise ValueError("i_max must be positive")
        if a <= 0 or a > 2:
            raise ValueError("a upper bound is 2 and lower bound is 0")

    def _independent_estimator(self, values: np._typing.NDArray) -> float:
        """Apply function to row of matrix and find mean of row

        Args:
            values: row of random values matrix

        Returns: mean of row

        """
        vfunc = np.vectorize(self.func)
        return 1 / len(values) * np.sum(vfunc(values))

    def _estimator(self, random_matrix: np._typing.NDArray) -> tuple[float, np._typing.NDArray]:
        """Find mean of all rows

        Args:
            random_matrix: matrix of random values

        Returns: mean of all rows

        """
        values = np.array(list(map(self._independent_estimator, random_matrix)))
        return 1 / self.count * np.sum(values), values

    def _update_independent_estimator(self, i: int, old_value: float, new_values: np._typing.NDArray) -> float:
        """Update mean of row

        Args:
            i: step count
            old_value: previous value of row on i-1 step
            new_values: new generated row of random values

        Returns: Updated mean of row

        """

        return (i * old_value + self._independent_estimator(new_values[i * self.base_n :])) / (i + 1)

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
        sum_of_new: float = 0.0
        for i in range(self.count):
            old_value, new_values = old_values[i], random_matrix[i]
            value = self._update_independent_estimator(j, old_value, new_values)
            values.append(value)
            sum_of_new += value
        np_values = np.array(values)
        return (1 / self.count) * sum_of_new, np_values

    def _sigma(self, values: np._typing.NDArray, approximation: float) -> float:
        """Calculate parameter sigma for estimation

        Args:
            values:
            approximation:

        Returns: return sigma parameter

        """
        diff = np.sum(np.power(values - approximation, 2))
        return (1 / (self.count - 1) * diff) ** 0.5

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
            if current_error_tolerance.real < self.error_tolerance:
                return approximation, current_error_tolerance.real
            sobol_sampler.reset()
            sobol_sample = np.repeat(sobol_sampler.random(self.base_n * (i + 1)).transpose(), self.count, axis=0)
            sample = self._digital_shift(sobol_sample, xor_sample)
            approximation, values = self._update(i, values, sample)
            current_error_tolerance = self._sigma(values, approximation) * self.z
        return approximation, current_error_tolerance.real

    def _digital_shift(self, sobol_sequences: tpg.NDArray, xor_sample: tpg.NDArray) -> tpg.NDArray:
        """Digital shift of the sobol sequence

        Args:
            sobol_sequences: B Sobol sequences with length i*N
            xor_sample: Sample of Uniform distribution with length B

        Returns: XOR Sobol sequences with xor sample

        """

        def inner_func(sequence: tpg.NDArray, random_value: float) -> tpg.NDArray:
            """Xor sequence with random value

            Args:
                sequence: Sobol sequence of length i*N
                random_value: Random value from sample of Uniform distribution

            Returns: XOR sequence with random value

            """

            return np.array(list(map(lambda x: self._xor_float(x, random_value), sequence)))

        pair = list(zip(sobol_sequences, xor_sample))
        sobol_sequences = np.array(list(map(lambda x: inner_func(*x), pair)))

        return sobol_sequences

    @staticmethod
    def _xor_float(a: float, b: float) -> float:
        """XOR float values

        Args:
            a: First float value
            b: Second float value

        Returns: XOR float value

        """
        a = int(a * (2 ** BITS))
        b = int(b * (2 ** BITS))
        return np.bitwise_xor(a, b) / 2 ** BITS

    def __call__(self) -> tuple[float, float]:
        """Interface for users

        Returns: approximation for integral of function from 0 to 1

        """
        return self.rqmc()


class RQMCIntegrator:

    def compute_integral(self, func: Callable, params: dict) -> IntegrationResult:
        """
        Compute integral via RQMC integrator

        Args:
            func: integrated function
            params: Parameters of integration algorithm

        Returns: moment approximation and error tolerance
        """

        rqmc = RQMC(func, **params)()
        return IntegrationResult(rqmc[0], rqmc[1])
