from __future__ import annotations

import unittest

import numpy as np

from qmarg.basis import DisplacedHoBasis, hermite_phys
from qmarg.fock import (
    associated_laguerre,
    displaced_ho_overlap,
    displacement_matrix_element,
    hermite_shifted_coefficients,
    ho_gaussian_matrix_element,
    kinetic_matrix_element,
    shifted_gaussian_moment,
)
from qmarg.quadrature import GaussHermiteQuadrature


def integrate_on_real_line(values: np.ndarray, order: int = 220) -> float:
    quadrature = GaussHermiteQuadrature(order)
    x = quadrature.nodes()
    w = quadrature.weights_for_plain_integral()
    return float(np.sum(w * values(x)))


class FockKernelTest(unittest.TestCase):
    def test_associated_laguerre_known_values(self) -> None:
        x_values = [0.0, 0.4, 1.3]
        for x in x_values:
            self.assertAlmostEqual(associated_laguerre(0, 3, x), 1.0)
            self.assertAlmostEqual(associated_laguerre(1, 2, x), 3.0 - x)
            self.assertAlmostEqual(associated_laguerre(2, 0, x), 1.0 - 2.0 * x + 0.5 * x**2)

    def test_displacement_identity_at_zero_beta(self) -> None:
        for n in range(5):
            for m in range(5):
                expected = 1.0 if n == m else 0.0
                self.assertAlmostEqual(
                    displacement_matrix_element(n, m, 0.0),
                    expected,
                    places=14,
                )

    def test_displacement_real_symmetry(self) -> None:
        for beta in (-1.2, -0.3, 0.7, 1.5):
            for n in range(5):
                for m in range(5):
                    self.assertAlmostEqual(
                        displacement_matrix_element(n, m, beta),
                        displacement_matrix_element(m, n, -beta),
                        places=14,
                    )

    def test_hermite_shifted_coefficients_round_trip(self) -> None:
        omega = 0.8
        center = -0.45
        x = np.array([-1.2, -0.1, 0.0, 0.7, 1.4])
        for n in range(6):
            coeffs = hermite_shifted_coefficients(n, center, omega)
            reconstructed = np.polynomial.polynomial.polyval(x, coeffs)
            direct = hermite_phys(n, np.sqrt(omega) * (x - center))
            self.assertTrue(np.allclose(reconstructed, direct, atol=1e-13, rtol=1e-13))

    def test_shifted_gaussian_moment_low_orders(self) -> None:
        exponent = 0.7
        center = -0.4
        sqrt_pi_over_a = np.sqrt(np.pi / exponent)

        self.assertAlmostEqual(
            shifted_gaussian_moment(0, exponent, center),
            sqrt_pi_over_a,
            places=14,
        )
        self.assertAlmostEqual(
            shifted_gaussian_moment(1, exponent, center),
            center * sqrt_pi_over_a,
            places=14,
        )
        self.assertAlmostEqual(
            shifted_gaussian_moment(2, exponent, center),
            (center**2 + 1.0 / (2.0 * exponent)) * sqrt_pi_over_a,
            places=14,
        )

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

    def test_kinetic_matrix_element_is_hermitian(self) -> None:
        omega = 0.8
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=4, omega=omega)
        states = basis.states()

        for n, center_left in states:
            for m, center_right in states:
                left_right = kinetic_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    omega,
                )
                right_left = kinetic_matrix_element(
                    m,
                    center_right,
                    n,
                    center_left,
                    omega,
                )
                self.assertAlmostEqual(left_right, right_left, places=12)

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

    def test_centered_ho_gaussian_matrix_element_parity(self) -> None:
        omega = 0.8
        alpha = 0.35
        center = 0.0

        for n in range(5):
            for m in range(5):
                value = ho_gaussian_matrix_element(n, center, m, center, omega, alpha, center)
                if (n + m) % 2 == 1:
                    self.assertAlmostEqual(value, 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
