from __future__ import annotations

import unittest

import numpy as np

from qmarg.algebraic_assembler import AlgebraicGaussianExpansionAssembler
from qmarg.assembler import RealSpaceMatrixAssembler
from qmarg.basis import DisplacedHoBasis
from qmarg.experiments import ExperimentConfig, compare_basis_size, convergence_rows
from qmarg.fock import displaced_ho_overlap, displacement_matrix_element
from qmarg.problems import GaussianExpansionProblem, GaussianPotentialTerm
from qmarg.quadrature import GaussHermiteQuadrature


class SoftCoulombLcaoTest(unittest.TestCase):
    def test_displacement_closed_form_matches_overlap_convention(self) -> None:
        omega = 0.8
        center_left = -1.5
        center_right = 1.5
        beta = np.sqrt(omega / 2.0) * (center_right - center_left)

        self.assertAlmostEqual(
            displaced_ho_overlap(0, center_left, 1, center_right, omega),
            displacement_matrix_element(0, 1, beta),
        )

    def test_algebraic_gaussian_assembler_matches_real_space_assembler(self) -> None:
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=2, omega=0.8)
        problem = GaussianExpansionProblem(
            (
                GaussianPotentialTerm(coefficient=-0.7, exponent=0.35, center=-1.5),
                GaussianPotentialTerm(coefficient=-1.1, exponent=1.25, center=1.5),
            )
        )

        real_h, real_s = RealSpaceMatrixAssembler(GaussHermiteQuadrature(180)).assemble(
            problem, basis
        )
        alg_h, alg_s = AlgebraicGaussianExpansionAssembler().assemble(problem, basis)

        self.assertTrue(np.allclose(alg_s, real_s, atol=1e-11, rtol=1e-11))
        self.assertTrue(np.allclose(alg_h, real_h, atol=1e-11, rtol=1e-11))

    def test_truncated_prolog_backend_matches_real_space(self) -> None:
        """algebraic_truncated backend should agree with real_space within cutoff tolerance."""
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=2, omega=0.8)
        problem = GaussianExpansionProblem(
            (
                GaussianPotentialTerm(coefficient=-0.7, exponent=0.35, center=-1.5),
                GaussianPotentialTerm(coefficient=-1.1, exponent=1.25, center=1.5),
            )
        )

        quadrature = GaussHermiteQuadrature(180)
        real_h, real_s = RealSpaceMatrixAssembler(quadrature).assemble(problem, basis)
        trunc_h, trunc_s = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=12
        ).assemble(problem, basis)

        # Overlap matrix is exact (no truncation involved)
        self.assertTrue(np.allclose(real_s, trunc_s, atol=1e-12, rtol=1e-12))
        # Hamiltonian matrix difference is controlled by cutoff
        diff_h = float(np.max(np.abs(real_h - trunc_h)))
        self.assertLess(diff_h, 1e-3)

    def test_truncated_prolog_backend_higher_cutoff(self) -> None:
        """Higher cutoff should reduce the real_space vs truncated discrepancy."""
        basis = DisplacedHoBasis(center_distance=1.5, functions_per_center=2, omega=0.8)
        problem = GaussianExpansionProblem(
            (
                GaussianPotentialTerm(coefficient=-0.7, exponent=0.35, center=-1.5),
                GaussianPotentialTerm(coefficient=-1.1, exponent=1.25, center=1.5),
            )
        )

        quadrature = GaussHermiteQuadrature(180)
        real_h, _ = RealSpaceMatrixAssembler(quadrature).assemble(problem, basis)
        low_h, _ = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=4
        ).assemble(problem, basis)
        high_h, _ = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=14
        ).assemble(problem, basis)

        low_diff = float(np.max(np.abs(real_h - low_h)))
        high_diff = float(np.max(np.abs(real_h - high_h)))
        self.assertLess(high_diff, low_diff)

    def test_four_basis_comparison_runs(self) -> None:
        config = ExperimentConfig(grid_points=500, quadrature_order=120, candidate_count=9)
        summary = compare_basis_size(config, basis_size=4)

        self.assertEqual(len(summary.reference_eigenvalues), 2)
        self.assertEqual([model.basis_size for model in summary.models], [4, 4])
        for model in summary.models:
            self.assertTrue(np.all(np.isfinite(model.eigenvalues)))
            self.assertGreater(model.overlap_condition_number, 1.0)

    def test_eight_basis_improves_ground_state_over_two_basis(self) -> None:
        config = ExperimentConfig(grid_points=500, quadrature_order=120, candidate_count=9)
        rows = convergence_rows(config, basis_sizes=[2, 8])

        by_model = {(row.basis_size, row.model_name): row for row in rows}
        for prefix in ("displaced_ho", "monomial_gaussian_tower"):
            low = next(row for key, row in by_model.items() if key[0] == 2 and key[1].startswith(prefix))
            high = next(row for key, row in by_model.items() if key[0] == 8 and key[1].startswith(prefix))
            low_err = abs(low.eigenvalues[0] - low.reference_eigenvalues[0])
            high_err = abs(high.eigenvalues[0] - high.reference_eigenvalues[0])
            self.assertLess(high_err, low_err)


if __name__ == "__main__":
    unittest.main()
