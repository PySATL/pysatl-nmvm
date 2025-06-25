import math
import numpy as np

import pytest
from scipy.stats import expon, uniform

from estimators.semiparametric.nm_semiparametric_estimator import NMSemiParametricEstimator
from generators.nm_generator import NMGenerator
from mixtures.nm_mixture import NormalMeanMixtures


class TestSemiParametricSigmaEstimationEmpirical:
    generator = NMGenerator()

    @pytest.mark.parametrize(
        "real_sigma, sample_len, search_area", [(x, y, x + 5) for x in range(1, 10) for y in [1000, 5000]]
    )
    def test_sigma_estimation_empirical_expon(self, real_sigma: float, sample_len: int, search_area: float) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=expon)
        sample = self.generator.canonical_generate(mixture, sample_len)
        answer_list = []
        for alpha in [round(x, 5) for x in [i * 0.0001 for i in range(1, 10000)]]:
            t = math.sqrt(alpha * math.log(sample_len)) / (2 * search_area)
            estimator = NMSemiParametricEstimator("sigma_estimation_empirical", {"t": t})
            est = estimator.estimate(sample)
            left = (est.value**2 - real_sigma**2) ** 0.5
            right = (
                8
                * search_area**2
                / (alpha * math.log(sample_len) * sample_len ** ((1 - alpha) / 2) + 2 * math.exp(-1 * t))
            )
            answer_list.append(left < right)
        assert any(answer_list)

    @pytest.mark.parametrize("real_sigma, sample_len, search_area", [(1, 10000, 15)])
    def test_sigma_estimation_empirical_uniform(self, real_sigma: float, sample_len: int, search_area: float) -> None:
        mixture = NormalMeanMixtures("canonical", sigma=real_sigma, distribution=uniform)
        sample = self.generator.canonical_generate(mixture, sample_len)
        answer_list = []
        M = 10
        for alpha in [round(x, 4) for x in [i * 0.0001 for i in range(1, 10000)]]:
            t = math.sqrt(alpha * math.log(sample_len)) / (2 * search_area)
            estimator = NMSemiParametricEstimator("sigma_estimation_empirical", {"t": t})
            est = estimator.estimate(sample)
            left = abs(est.value**2 - real_sigma**2) ** 0.5
            right = 4 * M * search_area / math.sqrt(alpha * math.log(sample_len))
            if left < right:
                answer_list.append(est.value)
        print(min(answer_list))
        assert any(answer_list)
