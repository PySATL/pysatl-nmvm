from typing import Callable

from scipy.integrate import quad
from src.algorithms.support_algorithms.integrator import IntegrationResult

class QuadIntegrator:

    def compute_integral(self, func: Callable, params: dict) -> IntegrationResult:

        """
        Compute integral via quad integrator

        Args:
            func: integrated function
            params: Parameters of integration algorithm

        Returns: moment approximation and error tolerance
        """

        full_output_requested = params.pop('full_output', False)
        quad_res = quad(func, **params)
        if full_output_requested:
            value, error, message = quad_res
        else:
            value, error = quad_res
            message = None
        return IntegrationResult(value, error, message)
