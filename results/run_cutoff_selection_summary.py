"""
Cutoff selection summary: minimal cutoff satisfying accuracy targets.

Evaluates:
  max_abs_E_err < epsilon              (pointwise accuracy)
  max_delta_E < 2 * epsilon            (oscillation guard)

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

print("=== cutoff selection summary ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Gaussian expansion: {GAUSSIANS} functions")
print(f"Grid: fpc={FUNCTIONS_PER_CENTER_LIST}, cutoff={CUTOFFS}")
print(f"Targets: 1e-3, 1e-4")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

results = {}

for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis = DisplacedHoBasis(
        center_distance=A,
        functions_per_center=fpc,
        omega=OMEGA,
    )

    real_assembler = RealSpaceMatrixAssembler(quadrature)
    real_h, real_s = real_assembler.assemble(problem, basis)
    real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

    prev_e1 = None
    prev_e2 = None

    for cutoff in CUTOFFS:
        trunc_assembler = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=cutoff
        )
        trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
        trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

        abs_e1_err = abs(trunc_e[0] - real_e[0])
        abs_e2_err = abs(trunc_e[1] - real_e[1])
        max_abs_e_err = max(abs_e1_err, abs_e2_err)

        if prev_e1 is not None:
            delta_e1 = abs(trunc_e[0] - prev_e1)
            delta_e2 = abs(trunc_e[1] - prev_e2)
            max_delta_e = max(delta_e1, delta_e2)
        else:
            max_delta_e = None

        prev_e1 = trunc_e[0]
        prev_e2 = trunc_e[1]

        results.setdefault(fpc, []).append(
            {
                "cutoff": cutoff,
                "max_abs_e_err": max_abs_e_err,
                "max_delta_e": max_delta_e,
            }
        )

# Summary table
print("=" * 50)
print("Cutoff selection summary")
print("=" * 50)
print(f"{'fpc':>3s}  {'cutoff(1e-3)':>14s}  {'cutoff(1e-4)':>14s}")
print("-" * 36)

for fpc in FUNCTIONS_PER_CENTER_LIST:
    k3 = None
    k4 = None
    for r in results[fpc]:
        if r["cutoff"] == CUTOFFS[0]:
            continue
        if r["max_abs_e_err"] < 1e-3 and r["max_delta_e"] < 2e-3:
            if k3 is None:
                k3 = r["cutoff"]
        if r["max_abs_e_err"] < 1e-4 and r["max_delta_e"] < 2e-4:
            if k4 is None:
                k4 = r["cutoff"]

    s3 = str(k3) if k3 is not None else "N/A"
    s4 = str(k4) if k4 is not None else "N/A"
    print(f"{fpc:>3d}  {s3:>14s}  {s4:>14s}")

print("")
print("=== Notes ===")
print("- Eigenvalue convergence is oscillatory with cutoff.")
print("- Both absolute error and last-step delta are required:")
print("  max_abs_E_err < epsilon  (pointwise accuracy)")
print("  max_delta_E   < 2*epsilon (oscillation guard)")
print("- This is a conservative cutoff selection rule.")
print("- First cutoff (K=2) has no delta; check starts at K=4.")
print("- This is a manual diagnostic, not a CI test.")
