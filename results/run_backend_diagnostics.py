"""
Diagnose instability in algebraic_truncated eigenvalues at large basis size.

For selected (fpc, cutoff) pairs, examine:
  1. Eigenvalue spectrum of H_trunc vs H_real
  2. Condition number of H_trunc
  3. Sign structure / negative eigenvalues
  4. Whether the eigenvalue ordering changes

Manual experiment, not a CI test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "project"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmarg.assembler import RealSpaceMatrixAssembler
from qmarg.basis import DisplacedHoBasis
from qmarg.gaussian_fit import fit_soft_coulomb_gaussians
from qmarg.problems import GaussianExpansionProblem, two_center_gaussian_expansion_terms
from qmarg.quadrature import GaussHermiteQuadrature
from qmarg.solver import SymmetricOrthogonalizationSolver

Z = 1.0
A = 1.5
MU = 0.7
GAUSSIANS = 4
QUADRATURE_ORDER = 80
NUM_STATES = 2

OMEGA = 0.8

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()
real_assembler = RealSpaceMatrixAssembler(quadrature)

for fpc in [4, 6, 8]:
    print(f"{'=' * 70}")
    print(f"fpc = {fpc} (basis size = {fpc * 2})")
    print(f"{'=' * 70}")

    basis = DisplacedHoBasis(center_distance=A, functions_per_center=fpc, omega=OMEGA)
    real_h, real_s = real_assembler.assemble(problem, basis)

    eig_all = np.linalg.eigh(real_h)[0]
    cond_real = np.linalg.cond(real_h)

    print(f"  real_space H:  N={real_h.shape[0]}, "
          f"λ_min={eig_all[0]:+.6e}, λ_max={eig_all[-1]:+.6e}, "
          f"cond={cond_real:.2e}")
    print(f"    full spectrum: {', '.join(f'{v:+.4e}' for v in eig_all)}")
    print()

    for cutoff in [2, 4, 6, 8, 10, 12]:
        trunc_assembler = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=cutoff
        )
        trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)

        eig_trunc = np.linalg.eigh(trunc_h)[0]
        cond_trunc = np.linalg.cond(trunc_h)
        n_negative = int(np.sum(eig_trunc < 0))

        s_eig = np.linalg.eigh(trunc_s)[0]
        s_negative = int(np.sum(s_eig < -1e-12))

        trunc_e_solver = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues
        real_e_solver = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

        e1_err = trunc_e_solver[0] - real_e_solver[0]
        e2_err = trunc_e_solver[1] - real_e_solver[1]

        print(f"  K={cutoff:2d}:  λ∈[{eig_trunc[0]:+.6e}, {eig_trunc[-1]:+.6e}], "
              f"cond={cond_trunc:.2e}, neg_evals={n_negative}/{len(eig_trunc)}, "
              f"s_neg={s_negative}")
        print(f"            spectrum: {', '.join(f'{v:+.4e}' for v in eig_trunc)}")
        print(f"            solver E1/E2 = {trunc_e_solver[0]:+.6e} / {trunc_e_solver[1]:+.6e} "
              f"(err={e1_err:.2e} / {e2_err:.2e})")
        print()

print("")
print("=== Interpretation ===")
print("- neg_evals: number of negative eigenvalues in H_trunc.")
print("  The real-space Hamiltonian is positive definite (all eigenvalues > 0).")
print("  Negative eigenvalues indicate truncation has destroyed the matrix structure.")
print("- s_neg: number of negative overlap eigenvalues (should be 0).")
print("- cond: condition number. Large values indicate ill-conditioning.")
print("- If H_trunc has negative eigenvalues, the symmetric orthogonalization")
print("  solver may produce incorrect results for the lowest states.")
