from __future__ import annotations

import unittest

import numpy as np

from qmarg.basis import DisplacedHoBasis
from qmarg.fock import (
    displaced_ho_overlap,
    ho_gaussian_matrix_element,
    kinetic_matrix_element,
)
from qmarg.quadrature import GaussHermiteQuadrature


def integrate_on_real_line(values: np.ndarray, order: int = 220) -> float:
    quadrature = GaussHermiteQuadrature(order)
    x = quadrature.nodes()
    w = quadrature.weights_for_plain_integral()
    return float(np.sum(w * values(x)))


class FockKernelTest(unittest.TestCase):
    def test_displaced_overlap_matches_real_space_integral(self) -> None:
        omega = 0.8
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=3, omega=omega)
        states = basis.states()

        quadrature = GaussHermiteQuadrature(220)
        x = quadrature.nodes()
        w = quadrature.weights_for_plain_integral()
        phi = basis.values(x)

        for i, (n, center_left) in enumerate(states):
            for j, (m, center_right) in enumerate(states):
                real_space = float(np.sum(w * phi[i] * phi[j]))
                algebraic = displaced_ho_overlap(n, center_left, m, center_right, omega)
                self.assertAlmostEqual(algebraic, real_space, places=12)

    def test_kinetic_matrix_element_matches_real_space_integral(self) -> None:
        omega = 0.8
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=3, omega=omega)
        states = basis.states()

        quadrature = GaussHermiteQuadrature(220)
        x = quadrature.nodes()
        w = quadrature.weights_for_plain_integral()
        phi = basis.values(x)
        d2phi = basis.second_derivatives(x)

        for i, (n, center_left) in enumerate(states):
            for j, (m, center_right) in enumerate(states):
                real_space = float(np.sum(w * phi[i] * (-0.5 * d2phi[j])))
                algebraic = kinetic_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    omega,
                )
                self.assertAlmostEqual(algebraic, real_space, places=11)

    def test_ho_gaussian_matrix_element_matches_real_space_integral(self) -> None:
        omega = 0.8
        alpha = 0.35
        gaussian_center = 0.25
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=3, omega=omega)
        states = basis.states()

        quadrature = GaussHermiteQuadrature(220)
        x = quadrature.nodes()
        w = quadrature.weights_for_plain_integral()
        phi = basis.values(x)
        gaussian = np.exp(-alpha * (x - gaussian_center) ** 2)

        for i, (n, center_left) in enumerate(states):
            for j, (m, center_right) in enumerate(states):
                real_space = float(np.sum(w * phi[i] * gaussian * phi[j]))
                algebraic = ho_gaussian_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    omega,
                    alpha,
                    gaussian_center,
                )
                self.assertAlmostEqual(algebraic, real_space, places=12)


if __name__ == "__main__":
    unittest.main()
