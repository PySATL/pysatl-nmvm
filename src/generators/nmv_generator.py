"""Normal Mean-Variance (NMV) generator module.

This module provides generators for Normal Mean-Variance mixtures
in both classical and canonical forms.
"""

import numpy._typing as tpg
import scipy

from generators.abstract_generator import AbstractGenerator
from mixtures.abstract_mixture import AbstractMixtures
from mixtures.nmv_mixture import NormalMeanVarianceMixtures


class NMVGenerator(AbstractGenerator):

    @staticmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Classical form of NMVM

        Args:
            mixture: Normal Mean Variance Mixtures
            size: length of sample

        Returns: sample of given size

        Raises:
            ValueError: If mixture type is not Normal Mean Variance Mixtures

        """

        if not isinstance(mixture, NormalMeanVarianceMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.params.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return (
            mixture.params.alpha
            + mixture.params.beta * mixing_values
            + mixture.params.gamma * (mixing_values**0.5) * normal_values
        )

    @staticmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Canonical form of NMVM

        Args:
            mixture: Normal Mean Variance Mixtures
            size: length of sample

        Returns: sample of given size

        Raises:
            ValueError: If mixture type is not Normal Mean Variance Mixtures

        """

        if not isinstance(mixture, NormalMeanVarianceMixtures):
            raise ValueError("Mixture must be NormalMeanMixtures")
        mixing_values = mixture.params.distribution.rvs(size=size)
        normal_values = scipy.stats.norm.rvs(size=size)
        return mixture.params.alpha + mixture.params.mu * mixing_values + (mixing_values**0.5) * normal_values
