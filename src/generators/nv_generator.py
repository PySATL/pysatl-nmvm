import numpy._typing as tpg
import scipy

from src.generators.abstract_generator import AbstractGenerator
from src.mixtures.abstract_mixture import AbstractMixtures
from src.mixtures.nv_mixture import NormalVarianceMixtures


class NVGenerator(AbstractGenerator):

    @staticmethod
    def generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Classical form of NVM

        Args:
            mixture: Normal Variance Mixtures
            size: length of sample

        Returns: sample of given size

        Raises:
            ValueError: If mixture type is not Normal Variance Mixtures

        """

        if not isinstance(mixture, NormalVarianceMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.params.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        if mixture.mixture_form == "canonical":
            return mixture.params.alpha + (mixing_values ** 0.5) * normal_values
        return mixture.params.alpha + mixture.params.gamma * (mixing_values**0.5) * normal_values
