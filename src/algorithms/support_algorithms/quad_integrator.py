from typing import Callable, Any, Sequence

from scipy.integrate import quad
from src.algorithms.support_algorithms.integrator import IntegrationResult

class QuadIntegrator:

    def __init__(
            self,
            a: float = 0,
            b: float = 1,
            args: tuple[Any, ...] = (),
            full_output: int = 0,
            epsabs: float | int = 1.49e-08,
            epsrel: float | int = 1.49e-08,
            limit: float | int = 50,
            points: Sequence[float | int] | None = None,
            weight: float | int | None = None,
            wvar: Any = None,
            wopts: Any = None,
            maxp1: float | int = 50,
            limlst: int = 50,
            complex_func: bool = False,
    ):
        self.params = {
            'a': a,
            'b': b,
            'args': args,
            'full_output': full_output,
            'epsabs': epsabs,
            'epsrel': epsrel,
            'limit': limit,
            'points': points,
            'weight': weight,
            'wvar': wvar,
            'wopts': wopts,
            'maxp1': maxp1,
            'limlst': limlst,
            'complex_func': complex_func
        }

    def compute(self, func: Callable) -> IntegrationResult:

        """
        Compute integral via quad integrator

        Args:
            func: integrated function

        Returns: moment approximation and error tolerance
        """

        verbose = self.params.pop('full_output', False)
        result = quad(func, **self.params)
        if verbose:
            value, error, message = result
        else:
            value, error = result
            message = None
        return IntegrationResult(value, error, message)
