from dataclasses import dataclass
from typing import Any, Protocol, Callable, Optional


@dataclass
class IntegrationResult:
    value: float
    error: float
    message: Optional[dict[str, Any]] | None = None


class Integrator(Protocol):

    """Base class for integral calculation"""

    def compute_integral(self, func: Callable, params: dict) -> IntegrationResult:
        ...
