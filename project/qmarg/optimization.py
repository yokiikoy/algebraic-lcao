from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from qmarg.basis import BasisFunctionSet, DisplacedHoBasis, MonomialGaussianTowerBasis
from qmarg.domain import LcaoExperimentResult
from qmarg.problems import HamiltonianProblem
from qmarg.runner import LcaoRunner


class BasisFactory(Protocol):
    def create(self, parameter: float) -> BasisFunctionSet:
        ...


@dataclass(frozen=True)
class DisplacedHoBasisFactory:
    center_distance: float
    functions_per_center: int

    def create(self, parameter: float) -> BasisFunctionSet:
        return DisplacedHoBasis(
            center_distance=self.center_distance,
            functions_per_center=self.functions_per_center,
            omega=parameter,
        )


@dataclass(frozen=True)
class MonomialGaussianTowerBasisFactory:
    center_distance: float
    functions_per_center: int

    def create(self, parameter: float) -> BasisFunctionSet:
        return MonomialGaussianTowerBasis(
            center_distance=self.center_distance,
            functions_per_center=self.functions_per_center,
            gamma=parameter,
        )


def state_objective(eigenvalues: np.ndarray, mode: str) -> float:
    if mode == "e1":
        return float(eigenvalues[0])
    if mode == "e2":
        return float(eigenvalues[1])
    if mode == "sum12":
        return float(eigenvalues[0] + eigenvalues[1])
    raise ValueError(f"Unsupported objective mode: {mode}")


@dataclass(frozen=True)
class ScalarGridSearchOptimizer:
    candidates: Sequence[float]
    objective: str = "e1"

    def optimize(
        self,
        factory: BasisFactory,
        runner: LcaoRunner,
        problem: HamiltonianProblem,
        num_states: int,
    ) -> LcaoExperimentResult:
        best: LcaoExperimentResult | None = None
        best_score: float | None = None
        for candidate in self.candidates:
            result = runner.run(problem, factory.create(float(candidate)), num_states)
            score = state_objective(result.eigenvalues, self.objective)
            if best is None or best_score is None or score < best_score:
                best = result
                best_score = score

        if best is None:
            raise ValueError("No candidates provided.")
        return best
