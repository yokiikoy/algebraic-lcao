from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    left: float
    right: float
    num_points: int


@dataclass(frozen=True)
class EigenpairSummary:
    eigenvalues: np.ndarray
    overlap_condition_number: float


@dataclass(frozen=True)
class LcaoExperimentResult:
    model_name: str
    basis_size: int
    parameters: dict[str, float]
    eigenvalues: np.ndarray
    overlap_condition_number: float
