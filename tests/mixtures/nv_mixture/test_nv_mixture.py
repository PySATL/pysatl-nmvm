from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.stats import invgamma, t
from sklearn.metrics import mean_absolute_error

from src.mixtures.nv_mixture import NormalVarianceMixtures


def get_datasets(mixture_func, distribution_func, values):
    mixture_result, norm_result = np.vectorize(mixture_func)(values, {"error_tolerance": 0.001, "i_max": 300})[
        0
    ], np.vectorize(distribution_func)(values)
    return norm_result, mixture_result


def create_mixture_and_grid(params):
    nm_mixture = NormalVarianceMixtures(**params)
    values = np.linspace(-params["alpha"], params["alpha"], 40)
    return nm_mixture, values


def apply_params_grid(func_name, mix_and_distrib):
    result = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for mixture, distribution, values in mix_and_distrib:
            funcs = {
                "cdf": (mixture.compute_cdf, distribution.cdf),
                "pdf": (mixture.compute_pdf, distribution.pdf),
                "log": (mixture.compute_logpdf, distribution.logpdf),
                "moment": (mixture.compute_moment, distribution.moment),
            }
            if func_name == "moment":
                values = list(range(int(mixture.params.distribution.kwds["scale"])))
                mix_result = executor.submit(get_datasets, *funcs[func_name], values)
            else:
                mix_result = executor.submit(get_datasets, *funcs[func_name], values)
            result.append(mix_result)
    result = np.array([mean_absolute_error(*pair.result()) for pair in result])
    return result


class TestMultiVariateTDistribution:

    @pytest.fixture
    def generate_classic_distributions(self):
        grid_params = np.random.randint(4, 10, size=(1, 2))
        mix_and_distrib = []
        for params in grid_params:
            loc, v = params
            loc = 1
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "classical", "alpha": loc, "gamma": 1, "distribution": invgamma(v / 2, scale=v / 2)}
            )
            mult_t = t(df=v, loc=loc)
            mix_and_distrib.append((nm_mixture, mult_t, grid))
        return mix_and_distrib

    @pytest.fixture
    def generate_canonical_distributions(self):
        grid_params = np.random.randint(4, 10, size=(3, 2))
        mix_and_distrib = []
        for params in grid_params:
            loc, v = params
            loc = 1
            nm_mixture, grid = create_mixture_and_grid(
                {"mixture_form": "canonical", "alpha": loc, "distribution": invgamma(v / 2, scale=v / 2)}
            )
            mult_t = t(df=v, loc=loc)
            mix_and_distrib.append((nm_mixture, mult_t, grid))
        return mix_and_distrib

    def test_classic_cdf(self, generate_classic_distributions):
        result = apply_params_grid("cdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_cdf(self, generate_canonical_distributions):
        result = apply_params_grid("cdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_classic_pdf(self, generate_classic_distributions):
        result = apply_params_grid("pdf", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("pdf", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_classic_log_pdf(self, generate_classic_distributions):
        result = apply_params_grid("log", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_canonical_log_pdf(self, generate_canonical_distributions):
        result = apply_params_grid("log", generate_canonical_distributions)
        assert result.mean() < 1e-4

    def test_nm_classical_moment(self, generate_classic_distributions):
        result = apply_params_grid("moment", generate_classic_distributions)
        assert result.mean() < 1e-4

    def test_nm_canonical_moment(self, generate_canonical_distributions):
        result = apply_params_grid("moment", generate_canonical_distributions)
        assert result.mean() < 1e-4
