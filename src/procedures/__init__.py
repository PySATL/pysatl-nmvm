from src.procedures.semiparametric.nm_semi_param_algorithms.g_estimation_convolution import (
    NMSemiParametricGEstimation,
)
from src.procedures.semiparametric.nm_semi_param_algorithms.sigma_estimation_eigenvalue_based import (
    SemiParametricMeanSigmaEstimationEigenvalueBased,
)
from src.procedures.semiparametric.nm_semi_param_algorithms.sigma_estimation_empirical import (
    SemiParametricMeanSigmaEstimationEmpirical,
)
from src.procedures.semiparametric.nv_semi_param_algorithms.g_estimation_given_mu import (
    SemiParametricNVEstimation,
)
from src.procedures.semiparametric.nv_semi_param_algorithms.g_estimation_post_widder import (
    NVSemiParametricGEstimationPostWidder,
)
from src.procedures.semiparametric.nvm_semi_param_algorithms.g_estimation_given_mu import (
    SemiParametricGEstimationGivenMu,
)
from src.procedures.semiparametric.nvm_semi_param_algorithms.g_estimation_given_mu_rqmc_based import (
    SemiParametricGEstimationGivenMuRQMCBased,
)
from src.procedures.semiparametric.nvm_semi_param_algorithms.g_estimation_post_widder import (
    SemiParametricGEstimationPostWidder,
)
from src.procedures.semiparametric.nvm_semi_param_algorithms.mu_estimation import (SemiParametricMuEstimation)
from src.register.algorithm_purpose import AlgorithmPurpose
from src.register.register import Registry

ALGORITHM_REGISTRY: Registry = Registry()
ALGORITHM_REGISTRY.register("mu_estimation", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(SemiParametricMuEstimation)
ALGORITHM_REGISTRY.register("density_estim_inv_mellin_quad_int", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationGivenMu
)
ALGORITHM_REGISTRY.register("density_estim_deconv", AlgorithmPurpose.NM_SEMIPARAMETRIC)(NMSemiParametricGEstimation)
ALGORITHM_REGISTRY.register("sigma_estimation_eigenvalue", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    SemiParametricMeanSigmaEstimationEigenvalueBased
)
ALGORITHM_REGISTRY.register("sigma_estimation_laplace", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    SemiParametricMeanSigmaEstimationEmpirical
)
ALGORITHM_REGISTRY.register("density_estim_inv_mellin_rqmc_int", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationGivenMuRQMCBased
)
ALGORITHM_REGISTRY.register("density_estim_post_widder", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    SemiParametricGEstimationPostWidder
)
