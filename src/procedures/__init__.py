from src.procedures.semiparametric.nm_semiparametric.g_estimation_convolution import (
    NMEstimationDensityInvMT,
)
from src.procedures.semiparametric.nm_semiparametric.sigma_estimation_eigenvalue_based import (
    NMEstimationSigmaEigenvals,
)
from src.procedures.semiparametric.nm_semiparametric.sigma_estimation_empirical import (
    NMEstimationSigmaLaplace,
)
from src.procedures.semiparametric.nv_semiparametric.g_estimation_given_mu import (
    NVEstimationDensityInvMT,
)
from src.procedures.semiparametric.nv_semiparametric.g_estimation_post_widder import (
    NMVEstimationDensityPW,
)
from src.procedures.semiparametric.nvm_semiparametric.g_estimation_given_mu import (
    NMVEstimationDensityInvMTquad,
)
from src.procedures.semiparametric.nvm_semiparametric.g_estimation_given_mu_rqmc_based import (
    NMVEstimationDensityInvMTquadRQMCBased,
)
from src.procedures.semiparametric.nvm_semiparametric.g_estimation_post_widder import (
    NMVEstimationDensityPW,
)
from src.procedures.semiparametric.nvm_semiparametric.mu_estimation import (NMVEstimationMu)
from src.register.algorithm_purpose import AlgorithmPurpose
from src.register.register import Registry

ALGORITHM_REGISTRY: Registry = Registry()
ALGORITHM_REGISTRY.register("mu_estimation", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(NMVEstimationMu)
ALGORITHM_REGISTRY.register("density_estim_inv_mellin_quad_int", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    NMVEstimationDensityInvMTquad
)
ALGORITHM_REGISTRY.register("density_estim_deconv", AlgorithmPurpose.NM_SEMIPARAMETRIC)(NMEstimationDensityInvMT)
ALGORITHM_REGISTRY.register("sigma_estimation_eigenvalue", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    NMEstimationSigmaEigenvals
)
ALGORITHM_REGISTRY.register("sigma_estimation_laplace", AlgorithmPurpose.NM_SEMIPARAMETRIC)(
    NMEstimationSigmaLaplace
)
ALGORITHM_REGISTRY.register("density_estim_inv_mellin_rqmc_int", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    NMVEstimationDensityInvMTquadRQMCBased
)
ALGORITHM_REGISTRY.register("density_estim_post_widder", AlgorithmPurpose.NMV_SEMIPARAMETRIC)(
    NMVEstimationDensityPW
)
