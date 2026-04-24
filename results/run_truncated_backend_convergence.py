"""
Manual convergence experiment for the algebraic_truncated backend.

Compares real_space vs algebraic_truncated eigenvalue and matrix errors
at varying cutoff values (2, 4, 6, 8, 10, 12).

This is a manual experiment, not a CI test.
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
FUNCTIONS_PER_CENTER = 2

CUTOFFS = [2, 4, 6, 8, 10, 12]
# If runtime becomes too long, trim to [2, 4, 6, 8, 10]

print("=== algebraic_truncated backend convergence ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}")
print(f"Basis: DisplacedHO, functions_per_center={FUNCTIONS_PER_CENTER}, omega={OMEGA}")
print(f"Gaussian expansion: {GAUSSIANS} functions")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)
basis = DisplacedHoBasis(
    center_distance=A,
    functions_per_center=FUNCTIONS_PER_CENTER,
    omega=OMEGA,
)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

real_assembler = RealSpaceMatrixAssembler(quadrature)
real_h, real_s = real_assembler.assemble(problem, basis)
real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

print(f"{'cutoff':>6s}  {'E1_err':>16s}  {'abs_E1_err':>16s}  {'E2_err':>16s}  {'abs_E2_err':>16s}  {'max_H_err':>16s}  {'max_S_err':>16s}")
print("-" * 112)

results = {}

for cutoff in CUTOFFS:
    trunc_assembler = RealSpaceMatrixAssembler(
        quadrature, backend="algebraic_truncated", cutoff=cutoff
    )
    trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
    trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

    e1_err = trunc_e[0] - real_e[0]
    e2_err = trunc_e[1] - real_e[1]
    abs_e1_err = abs(e1_err)
    abs_e2_err = abs(e2_err)
    max_h_err = float(np.max(np.abs(trunc_h - real_h)))
    max_s_err = float(np.max(np.abs(trunc_s - real_s)))

    results[cutoff] = {"e1": trunc_e[0], "e2": trunc_e[1]}

    print(
        f"{cutoff:>6d}  {e1_err:>+16.10e}  {abs_e1_err:>16.10e}  "
        f"{e2_err:>+16.10e}  {abs_e2_err:>16.10e}  "
        f"{max_h_err:>16.10e}  {max_s_err:>16.10e}"
    )

# Oscillatory convergence estimate using last two cutoffs
K_max = CUTOFFS[-1]
K_prev = CUTOFFS[-2]
r_max = results[K_max]
r_prev = results[K_prev]

E1_est = 0.5 * (r_max["e1"] + r_prev["e1"])
err1_est = abs(r_max["e1"] - r_prev["e1"])
E2_est = 0.5 * (r_max["e2"] + r_prev["e2"])
err2_est = abs(r_max["e2"] - r_prev["e2"])

real_e1, real_e2 = real_e[0], real_e[1]
print("")
print("=== Oscillatory convergence estimate (last two cutoffs) ===")
print(f"Real E1 = {real_e1:>+16.10e}")
print(f"Real E2 = {real_e2:>+16.10e}")
print(f"E1_est  = {E1_est:>+16.10e}  err_est = {err1_est:>16.10e}  (true err = {E1_est - real_e1:>+16.10e})")
print(f"E2_est  = {E2_est:>+16.10e}  err_est = {err2_est:>16.10e}  (true err = {E2_est - real_e2:>+16.10e})")
print(f"Note: signed errors may oscillate with cutoff; absolute errors and")
print(f"last-two-cutoff spread provide a conservative convergence bracket.")
