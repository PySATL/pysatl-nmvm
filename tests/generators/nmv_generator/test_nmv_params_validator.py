import pytest
from scipy.stats import norm

from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures


@pytest.mark.ci
class TestGenerateParamsValidators:

    @pytest.mark.parametrize(
        "params",
        [
            {"wrong_alpha": 1, "beta": 1, "gamma": 1, "distribution": norm},
            {"alpha": 1, "wrong_beta": 1, "gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "wrong_gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "gamma": 1, "wrong_distribution": norm},
        ],
    )
    def test_classical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalMeanVarianceMixtures("classical", **params)

    def test_classical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalMeanVarianceMixtures("classical", **{"alpha": 1, "beta": 1, "gamma": 1, "distribution": 1})

    @pytest.mark.parametrize(
        "params",
        [
            {"wrong_alpha": 1, "mu": 1, "distribution": norm},
            {"alpha": 1, "mu": 1, "wrong_distribution": norm},
            {"alpha": 1, "mu_wrong": 1, "distribution": norm},
        ],
    )
    def test_canonical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalMeanVarianceMixtures("canonical", **params)

    def test_canonical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalMeanVarianceMixtures("canonical", **{"alpha": 1, "mu": 1, "distribution": 1})
