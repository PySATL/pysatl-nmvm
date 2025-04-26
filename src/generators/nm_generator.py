import numpy._typing as tpg
import scipy

from src.generators.abstract_generator import AbstractGenerator
from src.mixtures.abstract_mixture import AbstractMixtures
from src.mixtures.nm_mixture import NormalMeanMixtures


class NMGenerator(AbstractGenerator):

    @staticmethod
    def generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Classical form of NMM

        Args:
            mixture: Normal Mean Mixture
            size: length of sample

        Returns: sample of given size

        Raises:
            ValueError: If mixture is not a Normal Mean Mixture

        """

        if not isinstance(mixture, NormalMeanMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.params.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        if mixture.mixture_form=="canonical":
            return mixing_values + mixture.params.sigma * normal_values
        return mixture.params.alpha + mixture.params.beta * mixing_values + mixture.params.gamma * normal_values
