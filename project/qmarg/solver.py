from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qmarg.domain import EigenpairSummary


class GeneralizedEigenSolver(Protocol):
    def solve(self, h: np.ndarray, s: np.ndarray, num_states: int) -> EigenpairSummary:
        ...


@dataclass(frozen=True)
class SymmetricOrthogonalizationSolver:
    overlap_threshold: float = 1e-10

    def solve(self, h: np.ndarray, s: np.ndarray, num_states: int) -> EigenpairSummary:
        evals_s, vecs_s = np.linalg.eigh(s)
        keep = evals_s > self.overlap_threshold
        if not np.any(keep):
            raise ValueError("Overlap matrix is numerically singular.")

        cond = float(evals_s[keep].max() / evals_s[keep].min())
        x = vecs_s[:, keep] @ np.diag(evals_s[keep] ** -0.5) @ vecs_s[:, keep].T
        h_ortho = x.T @ h @ x
        evals, _ = np.linalg.eigh(h_ortho)
        return EigenpairSummary(
            eigenvalues=evals[:num_states],
            overlap_condition_number=cond,
        )
