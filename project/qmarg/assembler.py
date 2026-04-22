from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qmarg.basis import BasisFunctionSet
from qmarg.problems import HamiltonianProblem
from qmarg.quadrature import RealLineQuadrature


class MatrixAssembler(Protocol):
    def assemble(
        self,
        problem: HamiltonianProblem,
        basis: BasisFunctionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class RealSpaceMatrixAssembler:
    quadrature: RealLineQuadrature

    def assemble(
        self,
        problem: HamiltonianProblem,
        basis: BasisFunctionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.quadrature.nodes()
        w = self.quadrature.weights_for_plain_integral()

        phi = basis.values(x)
        d2phi = basis.second_derivatives(x)
        v = problem.potential().value(x)

        dim = basis.size()
        s = np.zeros((dim, dim))
        h = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                s[i, j] = np.sum(w * phi[i] * phi[j])
                h[i, j] = np.sum(w * phi[i] * (-0.5 * d2phi[j] + v * phi[j]))

        return 0.5 * (h + h.T), 0.5 * (s + s.T)
