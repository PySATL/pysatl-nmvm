from typing import Callable

import numpy as np
import pytest

from src.algorithms.support_algorithms.rqmc import RQMC


def loss_func(true_func: Callable, rqms: Callable, count: int):
    true_value = true_func(1) - true_func(0)
    sum_of_diff = 0
    for _ in range(count):
        sum_of_diff += abs(true_value - rqms()[0])
    return 1 / count * sum_of_diff


class TestSimplyFunctions:
    error_tolerance = 1e-5

    def test_constant_func(self):
        rqmc = RQMC(lambda x: 1, error_tolerance=self.error_tolerance)
        assert loss_func(lambda x: x, rqmc.rqmc, 1000) < self.error_tolerance

    def test_linear_func(self):
        rqmc = RQMC(lambda x: x, error_tolerance=self.error_tolerance, i_max=300)
        assert loss_func(lambda x: np.power(x, 2) / 2, rqmc.rqmc, 100) < self.error_tolerance

    def test_polynom_func(self):
        rqmc = RQMC(lambda x: x**3 - x**2 + 1, error_tolerance=self.error_tolerance)
        assert loss_func(lambda x: (x**4) / 4 - (x**3) / 3 + x, rqmc.rqmc, 100) < self.error_tolerance


class TestHardFunctions:
    error_tolerance = 1e-4

    def test_trigonometric_func(self):
        rqmc = RQMC(lambda x: np.sin(x) + np.cos(x), error_tolerance=self.error_tolerance, i_max=100)
        assert loss_func(lambda x: np.sin(x) - np.cos(x), rqmc.rqmc, 100) < self.error_tolerance

    def test_mix_function(self):
        rqmc = RQMC(
            lambda x: (x / np.sin(x)) + (np.exp(-x) / np.cos(x)), error_tolerance=self.error_tolerance, i_max=100
        )
        assert loss_func(lambda x: 1.79789274334 if x == 1 else 0, rqmc.rqmc, 100)

    def test_log_function(self):
        rqmc = RQMC(
            lambda x: np.sign(x - 0.5) * abs(np.log(abs(x - 0.5))), error_tolerance=self.error_tolerance, i_max=100
        )
        assert loss_func(lambda x: 0, rqmc.rqmc, 100)


@pytest.mark.ci
class TestArgsParse:
    @pytest.mark.parametrize(
        "args",
        [
            (-1, 1, 2, 1, 1),
            (1, -1, 2, 1, 1),
            (1, 1, -1, 1, 1),
            (1, 1, 2, -1, 1),
            (1, 1, 1, 1, -1),
            (1, 1, 3, 1, 1),
            (1, 1, 2, 1, 10),
        ],
    )
    def test_args_parse(self, args):
        with pytest.raises(ValueError):
            RQMC._args_validation(*args)
