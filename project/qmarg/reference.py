from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qmarg.domain import GridSpec
from qmarg.problems import HamiltonianProblem


class ReferenceSolver(Protocol):
    def solve(self, problem: HamiltonianProblem, num_states: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class FiniteDifferenceReferenceSolver:
    grid: GridSpec

    def solve(self, problem: HamiltonianProblem, num_states: int) -> np.ndarray:
        x = np.linspace(self.grid.left, self.grid.right, self.grid.num_points)
        dx = x[1] - x[0]
        v = problem.potential().value(x)

        main = np.full(self.grid.num_points, 1.0 / dx**2) + v
        off = np.full(self.grid.num_points - 1, -0.5 / dx**2)
        h = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        evals, _ = np.linalg.eigh(h)
        return evals[:num_states]
