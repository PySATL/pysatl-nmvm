import math

import numpy as np
import pytest
from scipy.stats import expon, uniform

from src.estimators.semiparametric.nm_semiparametric_estimator import NMSemiParametricEstimator
from src.generators.nm_generator import NMGenerator
from src.mixtures.nm_mixture import NormalMeanMixtures


class TestSemiParametricSigmaEstimationEigenvalueBased:
    generator = NMGenerator()

    @pytest.mark.parametrize("real_sigma, sample_len, search_area, a, b", [(10, 1000, 20.0, 1 / 16, 1 / 8)])
    def test_sigma_estimation_eigenvalue_based_expon_single(
        self, real_sigma: float, sample_len: int, search_area: float, a: float, b: float
    ) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=expon)
        sample = self.generator.canonical_generate(mixture, sample_len)
        k = sample_len ** (1 / 2 - a - b)
        l = 1 / search_area * math.sqrt((a * math.log(sample_len)) / 2) * k
        eps = (
            math.sqrt(2 * a)
            / search_area
            * math.sqrt(math.log(sample_len))
            / sample_len**b
            * (np.log(np.log(sample_len))) ** 0.6
        )

        estimator = NMSemiParametricEstimator(
            "sigma_estimation_eigenvalue_based", {"k": k, "l": l, "eps": eps, "search_area": search_area}
        )
        est = estimator.estimate(sample)
        print(est.value)
        assert (est.value - real_sigma) ** 2 < real_sigma * 0.1

    @pytest.mark.parametrize(
        "real_sigma, sample_len, search_area, search_density, a, b",
        [(x, 10000, 4.0, 10000, 1 / 16, 1 / 8) for x in range(1, 3)],
    )
    def test_sigma_estimation_eigenvalue_based_expon_1(
        self, real_sigma: float, sample_len: int, search_area: float, search_density: float, a: float, b: float
    ) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=expon)
        sample = self.generator.canonical_generate(mixture, sample_len)
        k = sample_len ** (1 / 2 - a - b)
        l = 1 / search_area * math.sqrt((a * math.log(sample_len)) / 2) * k
        eps = (
            math.sqrt(2 * a)
            / search_area
            * math.sqrt(math.log(sample_len))
            / sample_len**b
            * (np.log(np.log(sample_len))) ** 0.6
        )

        estimator = NMSemiParametricEstimator(
            "sigma_estimation_eigenvalue_based", {"k": k, "l": l, "eps": eps, "search_area": search_area}
        )
        est = estimator.estimate(sample)
        print(est.value)
        assert abs(est.value - real_sigma) < real_sigma * 2

    @pytest.mark.parametrize(
        "real_sigma, sample_len, search_area, search_density, a, b",
        [(x, 10000, 20.0, 10000, 1 / 16, 1 / 8) for x in range(10, 15)],
    )
    def test_sigma_estimation_eigenvalue_based_expon_2(
        self, real_sigma: float, sample_len: int, search_area: float, search_density: float, a: float, b: float
    ) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=expon)
        sample = self.generator.canonical_generate(mixture, sample_len)
        k = sample_len ** (1 / 2 - a - b)
        l = 1 / search_area * math.sqrt((a * math.log(sample_len)) / 2) * k
        eps = (
            math.sqrt(2 * a)
            / search_area
            * math.sqrt(math.log(sample_len))
            / sample_len**b
            * (np.log(np.log(sample_len))) ** 0.6
        )

        estimator = NMSemiParametricEstimator(
            "sigma_estimation_eigenvalue_based", {"k": k, "l": l, "eps": eps, "search_area": search_area}
        )
        est = estimator.estimate(sample)
        print(est.value)
        assert abs(est.value - real_sigma) < real_sigma * 0.3

    @pytest.mark.parametrize(
        "real_sigma, sample_len, search_area, search_density, a, b",
        [(10, x, 20.0, 10000, 1 / 16, 1 / 8) for x in [1000, 5000, 10000, 100000]],
    )
    def test_sigma_estimation_eigenvalue_based_expon_3(
        self, real_sigma: float, sample_len: int, search_area: float, search_density: float, a: float, b: float
    ) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=expon)
        sample = self.generator.canonical_generate(mixture, sample_len)
        k = sample_len ** (1 / 2 - a - b)
        l = 1 / search_area * math.sqrt((a * math.log(sample_len)) / 2) * k
        eps = (
            math.sqrt(2 * a)
            / search_area
            * math.sqrt(math.log(sample_len))
            / sample_len**b
            * (np.log(np.log(sample_len))) ** 0.6
        )

        estimator = NMSemiParametricEstimator(
            "sigma_estimation_eigenvalue_based", {"k": k, "l": l, "eps": eps, "search_area": search_area}
        )
        est = estimator.estimate(sample)
        print(est.value)
        assert abs(est.value - real_sigma) < real_sigma * 0.3

    @pytest.mark.parametrize(
        "real_sigma, sample_len, search_area, search_density, a, b", [(10, 100000, 20.0, 10000, 1 / 16, 1 / 8)]
    )
    def test_sigma_estimation_eigenvalue_based_uniform(
        self, real_sigma: float, sample_len: int, search_area: float, search_density: float, a: float, b: float
    ) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=uniform)
        sample = self.generator.canonical_generate(mixture, sample_len)
        k = sample_len ** (1 / 2 - a - b)
        l = 1 / search_area * math.sqrt((a * math.log(sample_len)) / 2) * k
        eps = (
            math.sqrt(2 * a)
            / search_area
            * math.sqrt(math.log(sample_len))
            / sample_len**b
            * (np.log(np.log(sample_len))) ** 0.6
        )

        estimator = NMSemiParametricEstimator(
            "sigma_estimation_eigenvalue_based", {"k": k, "l": l, "eps": eps, "search_area": search_area}
        )
        est = estimator.estimate(sample)
        print(est.value)
        assert abs(est.value - real_sigma) < real_sigma * 0.3
