from src.algorithms.semiparam_algorithms.nm_semi_param_algorithms.g_estimation_convolution import (
    NMSemiParametricGEstimation,
)
from src.algorithms.semiparam_algorithms.nvm_semi_param_algorithms.g_estimation_given_mu import (
    SemiParametricGEstimationGivenMu,
)
from src.algorithms.semiparam_algorithms.nvm_semi_param_algorithms.mu_estimation import SemiParametricMuEstimation
from src.register.algorithm_purpose import AlgorithmPurpose
from src.register.register import Registry

ALGORITHM_REGISTRY: Registry = Registry()
ALGORITHM_REGISTRY.register("mu_estimation", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(SemiParametricMuEstimation)
ALGORITHM_REGISTRY.register("g_estimation_given_mu", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationGivenMu
)
ALGORITHM_REGISTRY.register("g_estimation_convolution", AlgorithmPurpose.NM_SEMIPARAMETRIC)(NMSemiParametricGEstimation)
