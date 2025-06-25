import numpy as np
import pytest
from mpmath import ln
from scipy.stats import expon, gamma, uniform

from estimators.semiparametric.nmv_semiparametric_estimator import NMVSemiParametricEstimator
from generators.nmv_generator import NMVGenerator
from mixtures.nmv_mixture import NormalMeanVarianceMixtures


class TestPostWidder:

    @pytest.mark.parametrize(
        "mu, sigma, degree, sample_size",
        [(0.1, 1, 2, 10000), (0.1, 1, 3, 10000), (0.5, 2, 2, 10000), (1, 1, 2, 10000), (2, 2, 2, 10000)],
    )
    def test_post_widder_expon(self, mu, sigma, degree, sample_size) -> None:

        mixture = NormalMeanVarianceMixtures("classical", alpha=0, beta=mu, gamma=sigma, distribution=expon)
        sample = NMVGenerator().classical_generate(mixture, sample_size)
        x_data = np.linspace(0.5, 10.0, 30)

        estimator = NMVSemiParametricEstimator(
            "g_estimation_post_widder", {"x_data": x_data, "mu": mu, "sigma": sigma, "n": degree}
        )
        est = estimator.estimate(sample)
        est_data = est.list_value
        error = [((1 / sample_size) * (est_data[i] - expon.pdf(x_data[i])) ** 2) ** 0.5 for i in range(len(x_data))]
        assert all([err < ln(ln(sample_size)) / ln(sample_size) for err in error])

    @pytest.mark.parametrize(
        "mu, sigma, degree, sample_size, a",
        [
            (0.1, 1, 2, 10000, 1),
            (0.1, 1, 3, 10000, 2),
            (0.5, 2, 2, 10000, 3),
            (1, 1, 2, 10000, 0.5),
            (2, 2, 2, 10000, 1),
        ],
    )
    def test_post_widder_gamma(self, mu, sigma, degree, sample_size, a) -> None:
        mixture = NormalMeanVarianceMixtures("classical", alpha=0, beta=mu, gamma=sigma, distribution=gamma(a))
        sample = NMVGenerator().classical_generate(mixture, sample_size)
        x_data = np.linspace(0.5, 10.0, 30)

        estimator = NMVSemiParametricEstimator(
            "g_estimation_post_widder", {"x_data": x_data, "mu": mu, "sigma": sigma, "n": degree}
        )
        est = estimator.estimate(sample)
        est_data = est.list_value
        error = [((1 / sample_size) * (est_data[i] - gamma(a).pdf(x_data[i])) ** 2) ** 0.5 for i in range(len(x_data))]
        assert all([err < ln(ln(sample_size)) / ln(sample_size) for err in error])
