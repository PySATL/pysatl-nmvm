import math

import pytest
from scipy.stats import expon, gamma, halfnorm, pareto

from src.estimators.semiparametric.nmv_semiparametric_estimator import NMVSemiparEstim
from src.generators.nmv_generator import NMVGenerator
from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures


class TestSemiParametricMuEstimation:
    generator = NMVGenerator()

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_expon_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("real_mu", [10**i for i in range(0, -10, -2)])
    def test_mu_estimation_expon_no_parameters_small(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_pareto_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=pareto(2.62))
        sample = self.generator.canonical_generate(mixture, 50000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("real_mu", [10**i for i in range(0, -10, -2)])
    def test_mu_estimation_pareto_no_parameters_small(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=pareto(2.62))
        sample = self.generator.canonical_generate(mixture, 50000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_halfnorm_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=halfnorm)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("real_mu", [i for i in range(-3, 3)])
    def test_mu_estimation_gamma_no_parameters(self, real_mu: float) -> None:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=gamma(2))
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation")
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("params", [{"m": m} for m in range(5, 10)])
    def test_mu_estimation_expon_1_parameter_m_positive(self, params: dict) -> None:
        real_mu = 1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("params", [{"m": m} for m in range(5, 10)])
    def test_mu_estimation_expon_1_parameter_m_negative(self, params: dict) -> None:
        real_mu = -1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params", [{"max_iterations": max_iterations} for max_iterations in (10**3, 10**4, 10**10)]
    )
    def test_mu_estimation_expon_1_parameter_max_iterations(self, params: dict) -> None:
        real_mu = 1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize("params", [{"m": m} for m in range(1, 5)])
    def test_mu_estimation_expon_1_parameter_m_is_best_estimation(self, params: dict) -> None:
        real_mu = 10
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(est.value == params["m"]) and est.success is False

    @pytest.mark.parametrize(
        "params", [{"m": 10, "tolerance": tol} for tol in (1 / 10**6, 1 / 10**7, 1 / 10**8, 1 / 10**9, 1 / 10**10)]
    )
    def test_mu_estimation_expon_2_parameters_tol_positive(self, params: dict) -> None:
        real_mu = 1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params", [{"m": 10, "tolerance": tol} for tol in (1 / 10**6, 1 / 10**7, 1 / 10**8, 1 / 10**9, 1 / 10**10)]
    )
    def test_mu_estimation_expon_2_parameters_tol_negative(self, params: dict) -> None:
        real_mu = -1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params",
        [
            {"m": 10, "tolerance": 10**-9, "omega": omega}
            for omega in [
                (lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0) for i in range(1, 20, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_omega_positive(self, params: dict) -> None:
        real_mu = 1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params",
        [
            {"m": 10, "tolerance": 10**-9, "omega": omega}
            for omega in [
                (lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0) for i in range(1, 20, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_omega_negative(self, params: dict) -> None:
        real_mu = -1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params",
        [
            {"m": m, "tolerance": tol, "omega": omega}
            for m in (10, 40, 70, 100)
            for tol in (1 / 10**6, 1 / 10**7, 1 / 10**8)
            for omega in [
                (lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0) for i in range(1, 6, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_all_positive(self, params: dict) -> None:
        real_mu = 1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True

    @pytest.mark.parametrize(
        "params",
        [
            {"m": m, "tolerance": tol, "omega": omega}
            for m in (10, 40, 70, 100)
            for tol in (1 / 10**6, 1 / 10**7, 1 / 10**8)
            for omega in [
                (lambda x: -1 * (x**i) * math.exp(-(1 / (1 - x**2))) if abs(x) < 1 else 0) for i in range(1, 6, 2)
            ]
        ],
    )
    def test_mu_estimation_expon_3_parameters_all_negative(self, params: dict) -> None:
        real_mu = -1
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=expon)
        sample = self.generator.canonical_generate(mixture, 10000)
        estimator = NMVSemiparEstim("mu_estimation", params)
        est = estimator.estimate(sample)
        assert abs(real_mu - est.value) < 1 and est.success is True
