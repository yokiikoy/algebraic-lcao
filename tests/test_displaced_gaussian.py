from __future__ import annotations

import unittest

from qmarg.fock import (
    displaced_gaussian_matrix_element,
    ho_gaussian_matrix_element,
    origin_gaussian_matrix_element,
    displacement_matrix_element,
)


class DisplacedGaussianTest(unittest.TestCase):
    def test_matches_ho_gaussian_matrix_element(self) -> None:
        """Direct comparison with the existing kernel."""
        import random
        random.seed(12345)
        # Sample a smaller but representative set of parameters
        for _ in range(20):
            omega = random.choice([0.6, 0.8, 1.2])
            alpha = random.choice([0.0, 0.15, 0.35, 0.9])
            center_left = random.choice([-0.7, 0.0, 0.5])
            center_right = random.choice([-0.3, 0.0, 0.9])
            gaussian_center = random.choice([-0.2, 0.0, 0.4])
            n = random.randrange(6)
            m = random.randrange(6)
            with self.subTest(
                omega=omega,
                alpha=alpha,
                A=center_left,
                B=center_right,
                C=gaussian_center,
                n=n,
                m=m,
            ):
                expected = ho_gaussian_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    omega,
                    alpha,
                    gaussian_center,
                )
                actual = displaced_gaussian_matrix_element(
                    n,
                    center_left,
                    m,
                    center_right,
                    omega,
                    alpha,
                    gaussian_center,
                )
                self.assertAlmostEqual(
                    actual, expected, places=10
                )

    def test_centered_limit(self) -> None:
        """When A = B = C = 0, reduces to origin_gaussian_matrix_element."""
        for omega in (0.6, 0.8, 1.2):
            for alpha in (0.15, 0.35, 0.9):
                for n in range(8):
                    for m in range(8):
                        with self.subTest(omega=omega, alpha=alpha, n=n, m=m):
                            expected = origin_gaussian_matrix_element(
                                n, m, omega, alpha
                            )
                            actual = displaced_gaussian_matrix_element(
                                n, 0.0, m, 0.0, omega, alpha, 0.0
                            )
                            self.assertAlmostEqual(actual, expected, places=12)

    def test_symmetric_when_centers_swap(self) -> None:
        """Swapping A <-> B should give same result for symmetric n,m?"""
        omega = 0.8
        alpha = 0.35
        A = -0.5
        B = 0.7
        C = 0.1
        for n in range(5):
            for m in range(5):
                val1 = displaced_gaussian_matrix_element(
                    n, A, m, B, omega, alpha, C
                )
                val2 = displaced_gaussian_matrix_element(
                    m, B, n, A, omega, alpha, C
                )
                self.assertAlmostEqual(val1, val2, places=12)

    def test_shifted_gaussian_only(self) -> None:
        """When A = B ≠ C, the formula should still work."""
        omega = 1.0
        alpha = 0.5
        center = 0.3
        gaussian_center = 0.7
        for n in range(5):
            for m in range(5):
                val = displaced_gaussian_matrix_element(
                    n, center, m, center, omega, alpha, gaussian_center
                )
                # Can compare with ho_gaussian_matrix_element
                expected = ho_gaussian_matrix_element(
                    n, center, m, center, omega, alpha, gaussian_center
                )
                self.assertAlmostEqual(val, expected, places=10)

    def test_midpoint_centered(self) -> None:
        """When C is midpoint of A and B, result should be symmetric."""
        omega = 0.9
        alpha = 0.4
        A = -0.6
        B = 0.6
        C = (A + B) / 2.0
        for n in range(5):
            for m in range(5):
                val = displaced_gaussian_matrix_element(
                    n, A, m, B, omega, alpha, C
                )
                # Compare with ho_gaussian_matrix_element
                expected = ho_gaussian_matrix_element(
                    n, A, m, B, omega, alpha, C
                )
                self.assertAlmostEqual(val, expected, places=10)

    def test_alpha_zero_identity(self) -> None:
        """α=0 → exp(-α(x-C)²) = identity operator."""
        omega = 0.8
        alpha = 0.0
        A = -0.4
        B = 0.3
        C = 0.1
        for n in range(5):
            for m in range(5):
                val = displaced_gaussian_matrix_element(
                    n, A, m, B, omega, alpha, C
                )
                # Should reduce to ⟨n,A|m,B⟩ = displacement matrix element
                beta = (B - A) * (omega / 2.0) ** 0.5
                expected = displacement_matrix_element(n, m, beta)
                self.assertAlmostEqual(val, expected, places=12)


    def test_convergence_with_increasing_cutoff(self) -> None:
        """Higher cutoff should converge closer to exact kernel."""
        omega = 0.8
        alpha = 0.35
        A = -0.5
        B = 0.7
        C = 0.1
        n, m = 2, 3

        reference = ho_gaussian_matrix_element(
            n, A, m, B, omega, alpha, C
        )

        errors = []
        for cutoff in (5, 10, 20, 40):
            approx = displaced_gaussian_matrix_element(
                n, A, m, B, omega, alpha, C, cutoff=cutoff
            )
            err = abs(approx - reference)
            errors.append(err)

        # Errors should generally decrease as cutoff increases
        # (allow small fluctuations by checking monotonic on average)
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], 1e-10)


if __name__ == "__main__":
    unittest.main()