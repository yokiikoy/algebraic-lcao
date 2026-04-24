"""
Manual convergence experiment for the algebraic_truncated backend.

Compares real_space vs algebraic_truncated eigenvalue and matrix errors
at varying cutoff values (2, 4, 6, 8).

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

CUTOFFS = [2, 4, 6, 8]

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

print(f"{'cutoff':>6s}  {'E1_err':>16s}  {'E2_err':>16s}  {'max_H_err':>16s}  {'max_S_err':>16s}")
print("-" * 76)

for cutoff in CUTOFFS:
    trunc_assembler = RealSpaceMatrixAssembler(
        quadrature, backend="algebraic_truncated", cutoff=cutoff
    )
    trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
    trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

    e1_err = trunc_e[0] - real_e[0]
    e2_err = trunc_e[1] - real_e[1]
    max_h_err = float(np.max(np.abs(trunc_h - real_h)))
    max_s_err = float(np.max(np.abs(trunc_s - real_s)))

    print(
        f"{cutoff:>6d}  {e1_err:>+16.10e}  {e2_err:>+16.10e}  "
        f"{max_h_err:>16.10e}  {max_s_err:>16.10e}"
    )
