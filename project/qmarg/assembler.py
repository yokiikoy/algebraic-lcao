from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from qmarg.basis import BasisFunctionSet
from qmarg.fock import (
    displaced_gaussian_matrix_element_truncated_prolog,
    displaced_ho_overlap,
    kinetic_matrix_element,
)
from qmarg.problems import GaussianExpansionProblem, HamiltonianProblem
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
    backend: Literal["real_space", "algebraic"] = "real_space"

    def assemble(
        self,
        problem: HamiltonianProblem,
        basis: BasisFunctionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "algebraic":
            return self._assemble_algebraic(problem, basis)

        return self._assemble_real_space(problem, basis)

    def _assemble_real_space(
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

    def _assemble_algebraic(
        self,
        problem: HamiltonianProblem,
        basis: BasisFunctionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(problem, GaussianExpansionProblem):
            raise TypeError(
                f"Algebraic backend requires a GaussianExpansionProblem, "
                f"got {type(problem).__name__}"
            )
        if not hasattr(basis, "omega") or not hasattr(basis, "states"):
            raise TypeError(
                f"Algebraic backend requires a basis with 'omega' and 'states()', "
                f"got {type(basis).__name__}"
            )

        dim = basis.size()
        states = basis.states()
        h = np.zeros((dim, dim))
        s = np.zeros((dim, dim))

        for i, (n, center_left) in enumerate(states):
            for j, (m, center_right) in enumerate(states):
                s[i, j] = displaced_ho_overlap(
                    n, center_left, m, center_right, basis.omega
                )
                h[i, j] = kinetic_matrix_element(
                    n, center_left, m, center_right, basis.omega
                )
                for term in problem.terms:
                    h[i, j] += term.coefficient * displaced_gaussian_matrix_element_truncated_prolog(
                        n, center_left, m, center_right,
                        basis.omega, term.exponent, term.center,
                    )

        return 0.5 * (h + h.T), 0.5 * (s + s.T)
