from __future__ import annotations

from dataclasses import dataclass

from qmarg.assembler import MatrixAssembler
from qmarg.basis import BasisFunctionSet
from qmarg.domain import LcaoExperimentResult
from qmarg.problems import HamiltonianProblem
from qmarg.solver import GeneralizedEigenSolver


@dataclass(frozen=True)
class LcaoRunner:
    assembler: MatrixAssembler
    solver: GeneralizedEigenSolver

    def run(
        self,
        problem: HamiltonianProblem,
        basis: BasisFunctionSet,
        num_states: int,
    ) -> LcaoExperimentResult:
        h, s = self.assembler.assemble(problem, basis)
        summary = self.solver.solve(h, s, num_states)
        return LcaoExperimentResult(
            model_name=basis.model_name(),
            basis_size=basis.size(),
            parameters=basis.parameter_dict(),
            eigenvalues=summary.eigenvalues,
            overlap_condition_number=summary.overlap_condition_number,
        )
