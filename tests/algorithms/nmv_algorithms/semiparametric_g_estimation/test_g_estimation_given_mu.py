import math

from scipy.stats import expon

from src.estimators.semiparametric.nmv_semiparametric_estimator import NMVSemiParametricEstimator
from src.generators.nmv_generator import NMVGenerator
from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures


class TestSemiParametricMixingDensityEstimationGivenMu:

    def test_g_estimation_expon(self) -> None:
        real_g = expon.pdf
        given_mu = 1
        n = 100

        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=given_mu, distribution=expon)
        sample = NMVGenerator().canonical_generate(mixture, n)
        print(sample)
        x_data = [0.5, 1, 3]
        estimator = NMVSemiParametricEstimator(
            "g_estimation_given_mu", {"x_data": x_data, "u_value": 7.6, "v_value": 0.9}
        )
        est = estimator.estimate(sample)
        error = 0.0
        for i in range(len(x_data)):
            error += math.sqrt(min(x_data[i], 1) * (est.list_value[i] - real_g(x_data[i])) ** 2)
            print(est.list_value[i], real_g(x_data[i]), error)
        error = error / len(x_data)
        assert error < n ** (-0.5)
