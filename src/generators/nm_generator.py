"""Normal Mean (NM) generator module.

This module provides generators for Normal Mean mixtures
in both classical and canonical forms.
"""

import numpy._typing as tpg
import scipy

from generators.abstract_generator import AbstractGenerator
from mixtures.abstract_mixture import AbstractMixtures
from mixtures.nm_mixture import NormalMeanMixtures


class NMGenerator(AbstractGenerator):

    @staticmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
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
        return mixture.params.alpha + mixture.params.beta * mixing_values + mixture.params.gamma * normal_values

    @staticmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Canonical form of NMM

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
        return mixing_values + mixture.params.sigma * normal_values
