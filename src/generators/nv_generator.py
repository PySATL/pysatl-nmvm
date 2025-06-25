"""Normal Variance (NV) generator module.

This module provides generators for Normal Variance mixtures
in both classical and canonical forms.
"""

import numpy._typing as tpg
import scipy

from generators.abstract_generator import AbstractGenerator
from mixtures.abstract_mixture import AbstractMixtures
from mixtures.nv_mixture import NormalVarianceMixtures


class NVGenerator(AbstractGenerator):

    @staticmethod
    def classical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
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
        return mixture.params.alpha + mixture.params.gamma * (mixing_values**0.5) * normal_values

    @staticmethod
    def canonical_generate(mixture: AbstractMixtures, size: int) -> tpg.NDArray:
        """Generate a sample of given size. Canonical form of NVM

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
        return mixture.params.alpha + (mixing_values**0.5) * normal_values
