"""Algorithms module for PySATL NMVM.

This module contains all algorithm implementations and provides a registry
for parametric and semiparametric estimation algorithms.
"""

from algorithms.semiparam_algorithms.nm_semi_param_algorithms.g_estimation_convolution import (
    NMSemiParametricGEstimation,
)
from algorithms.semiparam_algorithms.nm_semi_param_algorithms.sigma_estimation_eigenvalue_based import (
    SemiParametricMeanSigmaEstimationEigenvalueBased,
)
from algorithms.semiparam_algorithms.nm_semi_param_algorithms.sigma_estimation_empirical import (
    SemiParametricMeanSigmaEstimationEmpirical,
)
from algorithms.semiparam_algorithms.nv_semi_param_algorithms.g_estimation_given_mu import (
    SemiParametricNVEstimation,
)
from algorithms.semiparam_algorithms.nv_semi_param_algorithms.g_estimation_post_widder import (
    NVSemiParametricGEstimationPostWidder,
)
from algorithms.semiparam_algorithms.nvm_semi_param_algorithms.g_estimation_given_mu import (
    SemiParametricGEstimationGivenMu,
)
from algorithms.semiparam_algorithms.nvm_semi_param_algorithms.g_estimation_given_mu_rqmc_based import (
    SemiParametricGEstimationGivenMuRQMCBased,
)
from algorithms.semiparam_algorithms.nvm_semi_param_algorithms.g_estimation_post_widder import (
    SemiParametricGEstimationPostWidder,
)
from algorithms.semiparam_algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation
from register.algorithm_purpose import AlgorithmPurpose
from register.register import Registry

ALGORITHM_REGISTRY: Registry = Registry()
ALGORITHM_REGISTRY.register("mu_estimation", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(SemiParametricMuEstimation)
ALGORITHM_REGISTRY.register("g_estimation_given_mu", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationGivenMu
)
ALGORITHM_REGISTRY.register("g_estimation_convolution", AlgorithmPurpose.NM_SEMIPARAMETRIC)(NMSemiParametricGEstimation)
ALGORITHM_REGISTRY.register("sigma_estimation_eigenvalue_based", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    SemiParametricMeanSigmaEstimationEigenvalueBased
)
ALGORITHM_REGISTRY.register("sigma_estimation_empirical", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    SemiParametricMeanSigmaEstimationEmpirical
)
ALGORITHM_REGISTRY.register("g_estimation_given_mu_rqmc_based", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationGivenMuRQMCBased
)
ALGORITHM_REGISTRY.register("g_estimation_post_widder", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationPostWidder
)
