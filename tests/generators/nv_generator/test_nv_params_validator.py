import pytest
from scipy.stats import norm

from mixtures.nv_mixture import NormalVarianceMixtures


@pytest.mark.ci
class TestGenerateParamsValidators:
    @pytest.mark.parametrize(
        "params",
        [
            {"wrong_alpha": 1, "beta": 1, "distribution": norm},
            {"alpha": 1, "wrong_beta": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "wrong_distribution": norm},
        ],
    )
    def test_classical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalVarianceMixtures("classical", **params)

    def test_classical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalVarianceMixtures("classical", **{"alpha": 1, "beta": 1, "distribution": 1})

    @pytest.mark.parametrize("params", [{}, {"a": 1, "b": 2, "c": 3, "d": 4, "m": 5}])
    def test_canonical_args_validation_length_error(self, params):
        with pytest.raises(ValueError):
            NormalVarianceMixtures("canonical", **params)

    @pytest.mark.parametrize(
        "params", [{"wrong_alpha": 1, "distribution": norm}, {"alpha": 1, "wrong_distribution": norm}]
    )
    def test_canonical_wrong_names(self, params):
        with pytest.raises(ValueError):
            NormalVarianceMixtures("canonical", **params)

    def test_canonical_wrong_distribution_type(self):
        with pytest.raises(ValueError):
            NormalVarianceMixtures("canonical", **{"alpha": 1, "distribution": 1})
