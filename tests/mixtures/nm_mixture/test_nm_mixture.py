from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.stats import halfnorm, norm, skewnorm
from sklearn.metrics import mean_absolute_error

from src.mixtures.nm_mixture import NormalMeanMixtures, _NMMCanonicalDataCollector, _NMMClassicDataCollector


def create_mixture_and_grid(params):
    if params["mixture_form"] == "classical":
        nm_mixture = NormalMeanMixtures(**params)
        values = np.linspace(params["alpha"] + -3 * params["gamma"], params["alpha"] + 3 * params["gamma"], 40)
    else:
        nm_mixture = NormalMeanMixtures(**params)
        values = np.linspace(-3 * params["sigma"], 3 * params["sigma"], 40)
    return nm_mixture, values


def get_datasets(mixture_func, distribution_func, values):
    mixture_result, norm_result = np.vectorize(mixture_func)(values, {"error_tolerance": 0.001, "i_max": 300})[
        0
    ], np.vectorize(distribution_func)(values)
    return norm_result, mixture_result


def apply_params_grid(func_name, mix_and_distrib):
    result = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for mixture, distribution, values in mix_and_distrib:
            funcs = {
                "cdf": (mixture.compute_cdf, distribution.cdf),
                "pdf": (mixture.compute_pdf, distribution.pdf),
                "log": (mixture.compute_logpdf, distribution.logpdf),
            }
            mix_result = executor.submit(get_datasets, *funcs[func_name], values)
            result.append(mix_result)
    result = np.array([mean_absolute_error(*pair.result()) for pair in result])
    return result


@pytest.mark.ci
class TestParametersValidation:
    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"alpha": 0, "beta": 1, "gamma": 1, "distribution": norm, "some_unexpected_parameter": 3},
            {"alpha": 0, "beta": 1, "gamma": 1},
            {"alpha": [3], "beta": 1, "gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": [1], "gamma": 1, "distribution": norm},
            {"alpha": 1, "beta": 1, "gamma": norm, "distribution": norm},
            {"alpha": 1, "beta": 1, "gamma": 1, "distribution": 1},
        ],
    )
    def test_dataclass_creation_classic_form(self, params):
        with pytest.raises(ValueError):
            nm_mixture = NormalMeanMixtures("classical", **params)

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"sigma": 1, "distribution": norm, "some_unexpected_parameter": 3},
            {"sigma": 1},
            {"sigma": [3], "distribution": norm},
            {"sigma": 1, "distribution": 1},
        ],
    )
    def test_dataclass_creation_canonical_form(self, params):
        with pytest.raises(ValueError):
            nm_mixture = NormalMeanMixtures("canonical", **params)

    @pytest.mark.parametrize(
        "params",
        [
            {"mixture_form": "classical", "alpha": 0, "beta": 1, "gamma": 0, "distribution": norm},
            {"mixture_form": "canonical", "sigma": 0, "distribution": norm},
        ],
    )
    def test_mixture_params_validation(self, params):
        with pytest.raises(ValueError):
            nm_mixture = NormalMeanMixtures(**params)

    def test_dataclass_type(self):
        classic_mixture = NormalMeanMixtures("classical", alpha=0, beta=1, gamma=1, distribution=norm)
        canonical_mixture = NormalMeanMixtures("canonical", sigma=1, distribution=norm)
        assert isinstance(classic_mixture.params, _NMMClassicDataCollector) and isinstance(
            canonical_mixture.params, _NMMCanonicalDataCollector
        )


class TestNormalMeanMixturesBasicNormal:
    @pytest.fixture
    def generate_classic_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 3))
        mix_and_distrib = []
        for params in grid_params:
            alpha, beta, gamma = params
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "classical", "alpha": alpha, "beta": beta, "gamma": gamma, "distribution": norm}
            )
            parametric_norm = norm(alpha, np.sqrt(beta**2 + gamma**2))
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    @pytest.fixture
    def generate_canonical_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 1))
        mix_and_distrib = []
        for params in grid_params:
            sigma = params[0]
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "canonical", "sigma": sigma, "distribution": norm}
            )
            parametric_norm = norm(0, np.sqrt(1 + sigma**2))
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    def test_nm_classical_cdf(self, generate_classic_distributions):
        result = apply_params_grid("cdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_cdf(self, generate_canonical_distributions):
        result = apply_params_grid("cdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_nm_classical_pdf(self, generate_classic_distributions):
        result = apply_params_grid("pdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("pdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_nm_classical_log_pdf(self, generate_classic_distributions):
        result = apply_params_grid("log", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_log_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("log", generate_canonical_distributions)
        assert result.mean() < 1e-4


class TestNormalMeanMixtureNormal:
    @pytest.fixture
    def generate_classic_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 3))
        mix_and_distrib = []
        for params in grid_params:
            alpha, k, gamma = params
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "classical", "alpha": alpha, "beta": 1, "gamma": gamma, "distribution": norm(0, k)}
            )
            parametric_norm = norm(alpha, np.sqrt(k**2 + gamma**2))
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    @pytest.fixture
    def generate_canonical_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 2))
        mix_and_distrib = []
        for params in grid_params:
            k, sigma = params
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "canonical", "sigma": sigma, "distribution": norm(0, k)}
            )
            parametric_norm = norm(0, np.sqrt(k**2 + sigma**2))
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    def test_nm_classical_cdf(self, generate_classic_distributions):
        result = apply_params_grid("cdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_nm_canonical_cdf(self, generate_canonical_distributions):
        result = apply_params_grid("cdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_nm_classical_pdf(self, generate_classic_distributions):
        result = apply_params_grid("pdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_nm_canonical_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("pdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_nm_classical_log_pdf(self, generate_classic_distributions):
        result = apply_params_grid("log", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_nm_canonical_log_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("log", generate_canonical_distributions)
        assert result.mean() < 1e-4


class TestSkewNormalDistribution:
    @pytest.fixture
    def generate_classic_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 3))
        mix_and_distrib = []
        for params in grid_params:
            loc, scale, a = params
            delta = a / np.sqrt(1 + a**2)
            nm_mixture, grid = create_mixture_and_grid(
                {
                    "mixture_form": "classical",
                    "alpha": loc,
                    "beta": scale * delta,
                    "gamma": scale * np.sqrt(1 - delta**2),
                    "distribution": halfnorm(),
                }
            )
            parametric_norm = skewnorm(a=a, loc=loc, scale=scale)
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    @pytest.fixture
    def generate_canonical_distributions(self):
        grid_params = np.random.randint(1, 10, size=(3, 1))
        mix_and_distrib = []
        for params in grid_params:
            a = params[0]
            delta = a / np.sqrt(1 + a**2)
            scale = 1 / delta
            nm_mixture, grid = create_mixture_and_grid(
                {
                    "mixture_form": "canonical",
                    "sigma": scale * np.sqrt(1 - delta**2),
                    "distribution": halfnorm(),
                }
            )
            parametric_norm = skewnorm(a=a, loc=0, scale=scale)
            mix_and_distrib.append((nm_mixture, parametric_norm, grid))
        return mix_and_distrib

    def test_sk_classic_cdf(self, generate_classic_distributions):
        result = apply_params_grid("cdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_sk_canonical_cdf(self, generate_canonical_distributions):
        result = apply_params_grid("cdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_sk_classic_pdf(self, generate_classic_distributions):
        result = apply_params_grid("pdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_sk_canonical_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("pdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_sk_classic_log_pdf(self, generate_classic_distributions):
        result = apply_params_grid("log", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_sk_canonical_log_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("log", generate_canonical_distributions)
        assert result.mean() < 1e-4
