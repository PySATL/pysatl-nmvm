from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.stats import genhyperbolic, geninvgauss
from sklearn.metrics import mean_absolute_error

from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures


def get_datasets(mixture_func, distribution_func, values):
    mixture_result, norm_result = np.vectorize(mixture_func)(values, {"error_tolerance": 0.001})[0], np.vectorize(
        distribution_func
    )(values)
    return norm_result, mixture_result


def create_mixture_and_grid(params, p, a, b):
    nm_mixture = NormalMeanVarianceMixtures(**params)
    interval = genhyperbolic.interval(0.8, p, a, b)
    values = np.linspace(interval[0], interval[1], 10)
    return nm_mixture, values


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


class TestGeneralizedHyperbolicDistribution:
    @pytest.fixture
    def generate_classic_distributions(self):
        grid_params = np.random.uniform(1, 10, size=(3, 2))
        mix_and_distrib = []
        for params in grid_params:
            first, second = params
            beta = min(first, second)
            v = max(first, second)
            b = np.sqrt(v**2 - beta**2)
            nvm_mixture, grid = create_mixture_and_grid(
                {
                    "mixture_form": "classical",
                    "alpha": 0,
                    "beta": beta,
                    "gamma": 1,
                    "distribution": geninvgauss(p=1, b=b, loc=0, scale=1 / b),
                },
                1,
                v,
                beta,
            )
            gh = genhyperbolic(p=1, a=v, b=beta)
            mix_and_distrib.append((nvm_mixture, gh, grid))
        return mix_and_distrib

    @pytest.fixture
    def generate_canonical_distributions(self):
        grid_params = np.random.uniform(1, 10, size=(3, 2))
        mix_and_distrib = []
        for params in grid_params:
            first, second = params
            mu = min(first, second)
            v = max(first, second)
            b = np.sqrt(v**2 - mu**2)
            nvm_mixture, grid = create_mixture_and_grid(
                {
                    "mixture_form": "canonical",
                    "alpha": 0,
                    "mu": mu,
                    "distribution": geninvgauss(p=1, b=b, loc=0, scale=1 / b),
                },
                1,
                v,
                mu,
            )
            gh = genhyperbolic(p=1, a=v, b=mu)
            mix_and_distrib.append((nvm_mixture, gh, grid))
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
