"""
Error decomposition: separate basis error from cutoff error.

References:
  E_ref      = real_space eigenvalue at fpc=4 (largest basis, best estimate)
  E_real(fpc) = real_space eigenvalue at given fpc (no truncation)
  E_trunc(fpc,K) = truncated eigenvalue at (fpc, cutoff K)

Decomposition:
  total_error  = |E_trunc - E_ref|
  basis_error  = |E_real(fpc) - E_ref|     (constant across cutoffs)
  cutoff_error = |E_trunc - E_real(fpc)|   (decreases with cutoff)

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
FUNCTIONS_PER_CENTER_LIST = [1, 2, 3, 4]
CUTOFFS = [2, 4, 6, 8, 10, 12]

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()
real_assembler = RealSpaceMatrixAssembler(quadrature)

print("=== error decomposition: basis vs cutoff ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Reference: fpc=4, real_space")
print("")

real_e_by_fpc = {}
for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis = DisplacedHoBasis(center_distance=A, functions_per_center=fpc, omega=OMEGA)
    real_h, real_s = real_assembler.assemble(problem, basis)
    real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues
    real_e_by_fpc[fpc] = real_e

E1_ref = real_e_by_fpc[4][0]
E2_ref = real_e_by_fpc[4][1]

print(f"E1_ref (fpc=4, real) = {E1_ref:>+16.10e}")
print(f"E2_ref (fpc=4, real) = {E2_ref:>+16.10e}")
print("")

basis_by_fpc = {}
for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis_by_fpc[fpc] = DisplacedHoBasis(center_distance=A, functions_per_center=fpc, omega=OMEGA)

basis_e1_err = {fpc: abs(real_e_by_fpc[fpc][0] - E1_ref) for fpc in range(1, 4)}
basis_e2_err = {fpc: abs(real_e_by_fpc[fpc][1] - E2_ref) for fpc in range(1, 4)}

print("Basis error (real_space, no truncation):")
print(f"{'fpc':>3s}  {'basis_E1_err':>14s}  {'basis_E2_err':>14s}")
print("-" * 36)
for fpc in [1, 2, 3]:
    print(f"{fpc:>3d}  {basis_e1_err[fpc]:>14.4e}  {basis_e2_err[fpc]:>14.4e}")
print("")

print("=== Per-state decomposition ===")
for state_idx, state_name in enumerate(["E1", "E2"]):
    print(f"\n--- {state_name} ---")
    print(f"{'fpc':>3s}  {'cutoff':>6s}  {'total':>12s}  {'basis':>12s}  {'cutoff':>12s}  {'dominated by':>14s}")
    print("-" * 70)

    ref = E1_ref if state_idx == 0 else E2_ref

    for fpc in [1, 2, 3]:
        basis_err = basis_e1_err[fpc] if state_idx == 0 else basis_e2_err[fpc]

        for cutoff in CUTOFFS:
            trunc_assembler = RealSpaceMatrixAssembler(
                quadrature, backend="algebraic_truncated", cutoff=cutoff
            )
            trunc_h, trunc_s = trunc_assembler.assemble(problem, basis_by_fpc[fpc])
            trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

            cutoff_err = abs(trunc_e[state_idx] - real_e_by_fpc[fpc][state_idx])
            total_err = abs(trunc_e[state_idx] - ref)

            if basis_err > cutoff_err * 10:
                regime = "basis"
            elif cutoff_err > basis_err * 10:
                regime = "cutoff"
            else:
                regime = "mixed"

            print(
                f"{fpc:>3d}  {cutoff:>6d}  {total_err:>12.4e}  "
                f"{basis_err:>12.4e}  {cutoff_err:>12.4e}  {regime:>14s}"
            )

print("")
print("=== Interpretation ===")
print("- basis_error = |E_real(fpc) - E_ref| is the error floor: even perfect")
print("  truncation cannot beat it. Larger fpc reduces this floor.")
print("- cutoff_error = |E_trunc - E_real(fpc)| is the truncation error at")
print("  this basis. Increasing cutoff reduces it.")
print("- total_error = basis_error +~ cutoff_error (additive approximation).")
print("- Regime 'basis': basis error dominates, increasing cutoff won't help.")
print("- Regime 'cutoff': cutoff error dominates, increasing basis won't help.")
print("- Regime 'mixed': both contribute comparably, need to improve both.")
