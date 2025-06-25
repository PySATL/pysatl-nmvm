import math

import numpy as np
import pytest
from scipy.stats import expon, uniform

from estimators.semiparametric.nv_semiparametric_estimator import NVSemiParametricEstimator
from generators.nv_generator import NVGenerator
from mixtures.nv_mixture import NormalVarianceMixtures


class TestSemiParametricMixingDensityEstimationNV:

    @pytest.mark.parametrize("x_data", [np.linspace(0.1, 5, 10)])
    def test_g_estimation_expon(self, x_data) -> None:
        real_g = expon.pdf
        n = 100
        mixture = NormalVarianceMixtures("canonical", alpha=0, distribution=expon)
        sample = NVGenerator().canonical_generate(mixture, n)
        estimator = NVSemiParametricEstimator(
            "g_estimation_given_mu", {"x_data": x_data, "u_value": 7.6, "v_value": 0.9}
        )
        est = estimator.estimate(sample)
        error = 0.0
        for i in range(len(x_data)):
            error += math.sqrt(min(x_data[i], 1) * (est.list_value[i] - real_g(x_data[i])) ** 2)
        error = error / len(x_data)
        assert error < n ** (-0.5) * 2, f"Error {error} is too large"
