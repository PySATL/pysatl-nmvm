"""Estimate result data structure module.

This module defines the EstimateResult dataclass for storing
estimation results and metadata.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EstimateResult:
    value: float = -1
    list_value: List = field(default_factory=lambda: [-1])
    success: bool = False
    message: str = "No message"
