"""
Odd/even cutoff check: determine whether eigenvalue oscillation is parity-dependent.

Sweeps cutoff = [2,3,4,5,6,7,8,9,10,11,12] at fixed fpc=2
to verify that oscillation patterns are not artifacts of even-only sampling.

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
FPC = 2
CUTOFFS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

print("=== odd/even cutoff check ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Basis: fpc={FPC}")
print(f"Cutoff: {CUTOFFS}")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)
basis = DisplacedHoBasis(
    center_distance=A,
    functions_per_center=FPC,
    omega=OMEGA,
)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

real_assembler = RealSpaceMatrixAssembler(quadrature)
real_h, real_s = real_assembler.assemble(problem, basis)
real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

print(f"{'cutoff':>6s}  {'parity':>6s}  {'E1':>18s}  {'E2':>18s}  {'E1_err':>16s}  {'E2_err':>16s}")
print("-" * 93)

prev_e1 = None
prev_e2 = None

for cutoff in CUTOFFS:
    trunc_assembler = RealSpaceMatrixAssembler(
        quadrature, backend="algebraic_truncated", cutoff=cutoff
    )
    trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
    trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

    e1_err = trunc_e[0] - real_e[0]
    e2_err = trunc_e[1] - real_e[1]
    parity = "even" if cutoff % 2 == 0 else "odd"

    print(
        f"{cutoff:>6d}  {parity:>6s}  {trunc_e[0]:>+18.10e}  "
        f"{trunc_e[1]:>+18.10e}  {e1_err:>+16.10e}  {e2_err:>+16.10e}"
    )

    prev_e1 = trunc_e[0]
    prev_e2 = trunc_e[1]

print("")
print("=== Interpretation ===")
print("- If E1_err and E2_err show alternating sign only between even cutoffs,")
print("  oscillation is an even-step envelope effect (not parity-dependent).")
print("- If odd cutoffs break the sign-alternation pattern, the oscillation")
print("  is a genuine convergence phenomenon independent of parity.")
print("- Even-only sampling is sufficient for convergence studies.")
