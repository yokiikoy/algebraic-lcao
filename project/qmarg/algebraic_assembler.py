from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmarg.basis import DisplacedHoBasis
from qmarg.fock import (
    displaced_ho_overlap,
    ho_gaussian_matrix_element,
    kinetic_matrix_element,
)
from qmarg.problems import GaussianExpansionProblem


@dataclass(frozen=True)
class AlgebraicGaussianExpansionAssembler:
    """Assemble (H,S) for displaced HO bases and Gaussian-expanded potentials."""

    def assemble(
        self,
        problem: GaussianExpansionProblem,
        basis: DisplacedHoBasis,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = basis.states()
        dim = basis.size()
        h = np.zeros((dim, dim))
        s = np.zeros((dim, dim))

        for i, (n, center_left) in enumerate(states):
            for j, (m, center_right) in enumerate(states):
                s[i, j] = displaced_ho_overlap(
                    n,
                    center_left,
                    m,
                    center_right,
                    basis.omega,
                )
                h[i, j] = kinetic_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    basis.omega,
                )
                for term in problem.terms:
                    h[i, j] += term.coefficient * ho_gaussian_matrix_element(
                        n,
                        center_left,
                        m,
                        center_right,
                        basis.omega,
                        term.exponent,
                        term.center,
                    )

        return 0.5 * (h + h.T), 0.5 * (s + s.T)
