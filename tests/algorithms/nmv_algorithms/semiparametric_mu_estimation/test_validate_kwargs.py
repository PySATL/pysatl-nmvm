import math

import numpy as np
import pytest

from algorithms.semiparam_algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation


def _test_omega(x: float) -> float:
    return (-(x**3)) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0


@pytest.mark.ci
class TestValidateKwargs:
    @pytest.mark.parametrize(
        "params",
        [{"m": -10}, {"m": -0.5}, {"m": 0}, {"m": 0.5}, {"m": "str"}, {"m": []}, {"m": ()}],
    )
    def test_set_default_params_len_1_value_error_m(self, params: dict) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter m"):
            SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [{"tolerance": -1}, {"tolerance": -0.5}, {"tolerance": 0}, {"tolerance": []}, {"tolerance": "str"}],
    )
    def test_set_default_params_len_1_value_error_tolerance(self, params: dict) -> None:
        with pytest.raises(ValueError, match="Expected positive float as parameter tolerance"):
            SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize("params", [{"omega": 1}, {"omega": []}, {"omega": "str"}, {"omega": ()}])
    def test_set_default_params_len_3_value_error_omega(self, params: dict) -> None:
        with pytest.raises(ValueError, match="Expected callable object as parameter omega"):
            SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [
            {"max_iterations": -10},
            {"max_iterations": -0.5},
            {"max_iterations": 0},
            {"max_iterations": 0.5},
            {"max_iterations": "str"},
            {"max_iterations": []},
            {"max_iterations": ()},
        ],
    )
    def test_set_default_params_len_1_value_error_max_iterations(self, params: dict) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter max_iterations"):
            SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [
            {"m": -10, "tolerance": 1 / 10},
            {"m": -0.5, "tolerance": 1 / 10},
            {"m": 0, "tolerance": 1 / 10},
            {"m": 0.5, "tolerance": 1 / 10},
            {"m": [], "tolerance": 1 / 10},
            {"m": "str", "tolerance": 1 / 10},
        ],
    )
    def test_set_default_params_len_2_value_error_m(self, params: dict) -> None:
        with pytest.raises(ValueError, match="Expected positive integer as parameter m"):
            SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [
            {"m": 100, "tolerance": 1 / 10, "omega": _test_omega},
            {"m": 1000, "tolerance": 10**-9, "omega": _test_omega},
            {"m": 1, "tolerance": 1, "omega": _test_omega},
        ],
    )
    def test_set_default_params_len_3_correct(self, params: dict) -> None:
        SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [
            {"m": 100, "tolerance": 1 / 10, "omega": _test_omega, "max_iterations": 1000},
            {"m": 1000, "tolerance": 10**-9, "omega": _test_omega, "max_iterations": 100},
            {"m": 1, "tolerance": 1, "omega": _test_omega, "max_iterations": 10},
        ],
    )
    def test_set_default_params_len_4_correct(self, params: dict) -> None:
        SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params", [{"m": 100, "tolerance": 1 / 10}, {"m": 1000, "tolerance": 10**-9}, {"m": 1, "tolerance": 1}]
    )
    def test_set_default_params_len_2_correct(self, params: dict) -> None:
        SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize("params", [{"m": 100}, {"m": 1000}, {"m": 1}])
    def test_set_default_params_len_1_correct(self, params: dict) -> None:
        SemiParametricMuEstimation()._validate_kwargs(**params)

    @pytest.mark.parametrize(
        "params",
        [
            {"m": 100, "tolerance": 1 / 10, "omega": _test_omega},
            {"m": 1000, "tolerance": 10**-9, "omega": _test_omega},
            {"m": 1, "tolerance": 1, "omega": _test_omega},
        ],
    )
    def test_init_set_default_params_len_3_correct(self, params: dict) -> None:
        SemiParametricMuEstimation(np.array([1]), **params)

    @pytest.mark.parametrize(
        "params", [{"m": 100, "tolerance": 1 / 10}, {"m": 1000, "tolerance": 10**-9}, {"m": 1, "tolerance": 1}]
    )
    def test_init_set_default_params_len_2_correct(self, params: dict) -> None:
        SemiParametricMuEstimation(np.array([1]), **params)

    @pytest.mark.parametrize("params", [{"m": 100}, {"m": 1000}, {"m": 1}])
    def test_init_set_default_params_len_1_correct(self, params: dict) -> None:
        SemiParametricMuEstimation(np.array([1]), **params)
