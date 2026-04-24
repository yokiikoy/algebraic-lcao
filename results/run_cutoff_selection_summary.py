"""
Cutoff selection summary: practical rules from cutoff vs basis-size data.

Determines minimum safe cutoff for each (fpc, tolerance) using:
  1. max_abs_E_err < epsilon
  2. max_delta_E < 2 * epsilon

Manual diagnostic, not a CI test.
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
TOLERANCES = [1e-3, 1e-4]

print("=== cutoff selection summary ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Gaussian expansion: {GAUSSIANS} functions")
print(f"Grid: fpc={FUNCTIONS_PER_CENTER_LIST}, cutoff={CUTOFFS}")
print(f"Tolerances: {[f'{e:.0e}' for e in TOLERANCES]}")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()


def fmt_delta(val):
    return f"{val:>12.4e}" if val is not None else f"{'N/A':>12s}"


metrics_by_fpc = {}

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

        e1_err = trunc_e[0] - real_e[0]
        e2_err = trunc_e[1] - real_e[1]
        abs_e1_err = abs(e1_err)
        abs_e2_err = abs(e2_err)

        delta_e1 = abs(trunc_e[0] - prev_e1) if prev_e1 is not None else None
        delta_e2 = abs(trunc_e[1] - prev_e2) if prev_e2 is not None else None
        prev_e1 = trunc_e[0]
        prev_e2 = trunc_e[1]

        max_abs_e_err = max(abs_e1_err, abs_e2_err)
        max_delta_e = max(delta_e1, delta_e2) if delta_e1 is not None else None

        metrics_by_fpc.setdefault(fpc, []).append(
            {
                "cutoff": cutoff,
                "abs_e1_err": abs_e1_err,
                "abs_e2_err": abs_e2_err,
                "max_abs_e_err": max_abs_e_err,
                "delta_e1": delta_e1,
                "delta_e2": delta_e2,
                "max_delta_e": max_delta_e,
            }
        )

print(f"{'fpc':>3s}  {'cutoff':>6s}  {'abs_E1':>12s}  {'abs_E2':>12s}  "
      f"{'max_E':>12s}  {'delta_E1':>12s}  {'delta_E2':>12s}")
print("-" * 78)

for fpc in FUNCTIONS_PER_CENTER_LIST:
    for m in metrics_by_fpc[fpc]:
        print(
            f"{fpc:>3d}  {m['cutoff']:>6d}  {m['abs_e1_err']:>12.4e}  "
            f"{m['abs_e2_err']:>12.4e}  {m['max_abs_e_err']:>12.4e}  "
            f"{fmt_delta(m['delta_e1'])}  {fmt_delta(m['delta_e2'])}"
        )


def find_min_cutoff(data, eps):
    for m in data:
        if m["cutoff"] == CUTOFFS[0]:
            continue
        if m["max_abs_e_err"] < eps and m["max_delta_e"] < 2 * eps:
            return m["cutoff"]
    return None


print("")
print("=" * 58)
print("Cutoff selection summary")
print("=" * 58)
print(f"{'fpc':>3s}  {'cutoff(1e-3)':>14s}  {'cutoff(1e-4)':>14s}")
print("-" * 36)

for fpc in FUNCTIONS_PER_CENTER_LIST:
    data = metrics_by_fpc[fpc]
    k3 = find_min_cutoff(data, 1e-3)
    k4 = find_min_cutoff(data, 1e-4)
    s3 = str(k3) if k3 is not None else "N/A"
    s4 = str(k4) if k4 is not None else "N/A"
    print(f"{fpc:>3d}  {s3:>14s}  {s4:>14s}")

print("")
print("=== Notes ===")
print("- Oscillatory convergence requires using absolute error and delta.")
print("  Signed errors may cross zero accidentally; max_delta_E guards against")
print("  selecting a cutoff at a transient zero-crossing.")
print("- max_abs_E_err = max(|E1_err|, |E2_err|) checks real-space agreement.")
print("- max_delta_E = max(|E1(K)-E1(K-2)|, |E2(K)-E2(K-2)|) checks oscillation.")
print("- First cutoff (K=2) has no delta; acceptance check starts at K=4.")
print("- This is a manual diagnostic, not a CI test.")
