from __future__ import annotations

import unittest

from qmarg.fock import origin_gaussian_matrix_element
from qmarg.fock_su11 import origin_gaussian_matrix_element_su11


class FockSu11Test(unittest.TestCase):
    def test_origin_gaussian_su11_matches_reference_backend(self) -> None:
        for omega in (0.6, 0.8, 1.2):
            for alpha in (0.0, 0.15, 0.35, 0.9):
                for n in range(8):
                    for m in range(8):
                        with self.subTest(omega=omega, alpha=alpha, n=n, m=m):
                            self.assertAlmostEqual(
                                origin_gaussian_matrix_element_su11(n, m, omega, alpha),
                                origin_gaussian_matrix_element(n, m, omega, alpha),
                                places=12,
                            )

    def test_origin_gaussian_su11_parity(self) -> None:
        for omega, alpha in ((0.8, 0.35), (1.2, 0.9)):
            for n in range(8):
                for m in range(8):
                    if (n + m) % 2 == 1:
                        with self.subTest(omega=omega, alpha=alpha, n=n, m=m):
                            self.assertEqual(
                                origin_gaussian_matrix_element_su11(n, m, omega, alpha),
                                0.0,
                            )

    def test_origin_gaussian_su11_symmetry(self) -> None:
        for omega, alpha in ((0.8, 0.35), (1.2, 0.9)):
            for n in range(8):
                for m in range(8):
                    with self.subTest(omega=omega, alpha=alpha, n=n, m=m):
                        self.assertAlmostEqual(
                            origin_gaussian_matrix_element_su11(n, m, omega, alpha),
                            origin_gaussian_matrix_element_su11(m, n, omega, alpha),
                            places=14,
                        )


if __name__ == "__main__":
    unittest.main()
