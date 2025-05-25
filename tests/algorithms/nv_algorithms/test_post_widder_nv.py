import numpy as np
import pytest
from mpmath import ln
from scipy.stats import expon, gamma

from src.estimators.semiparametric.nv_semiparametric_estimator import NVSemiParametricEstimator
from src.generators.nv_generator import NVGenerator
from src.mixtures.nv_mixture import NormalVarianceMixtures


class TestPostWidderNV:

    @pytest.mark.parametrize(
        "sigma, degree, sample_size",
        [(1, 2, 10000), (1, 3, 10000), (2, 2, 10000), (1, 2, 10000), (2, 2, 10000)],
    )
    def test_post_widder_expon(self, sigma, degree, sample_size) -> None:

        mixture = NormalVarianceMixtures("classical", alpha=0, gamma=sigma, distribution=expon)
        sample = NVGenerator().classical_generate(mixture, sample_size)
        x_data = np.linspace(0.5, 10.0, 30)

        estimator = NVSemiParametricEstimator(
            "density_estim_post_widder", {"x_data": x_data, "sigma": sigma, "n": degree}
        )
        est = estimator.estimate(sample)
        est_data = est.list_value
        error = [((1 / sample_size) * (est_data[i] - expon.pdf(x_data[i])) ** 2) ** 0.5 for i in range(len(x_data))]
        assert all([err < ln(ln(sample_size)) / ln(sample_size) for err in error])

    @pytest.mark.parametrize(
        "sigma, degree, sample_size, a",
        [
            (1, 2, 10000, 1),
            (1, 3, 10000, 2),
            (2, 2, 10000, 3),
            (1, 2, 10000, 0.5),
            (2, 2, 10000, 1),
        ],
    )
    def test_post_widder_gamma(self, sigma, degree, sample_size, a) -> None:
        mixture = NormalVarianceMixtures("classical", alpha=0, gamma=sigma, distribution=gamma(a))
        sample = NVGenerator().classical_generate(mixture, sample_size)
        x_data = np.linspace(0.5, 10.0, 30)

        estimator = NVSemiParametricEstimator(
            "density_estim_post_widder", {"x_data": x_data, "sigma": sigma, "n": degree}
        )
        est = estimator.estimate(sample)
        est_data = est.list_value
        error = [((1 / sample_size) * (est_data[i] - gamma(a).pdf(x_data[i])) ** 2) ** 0.5 for i in range(len(x_data))]
        assert all([err < ln(ln(sample_size)) / ln(sample_size) for err in error])
