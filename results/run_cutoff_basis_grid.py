"""
Grid convergence experiment: cutoff vs basis size for algebraic_truncated backend.

Sweeps functions_per_center and cutoff to study their interaction.
Manual experiment, not a CI test.
"""

from __future__ import annotations

import sys
import time
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
FUNCTIONS_PER_CENTER_LIST = [1, 2, 3]
CUTOFFS = [2, 4, 6, 8, 10, 12]
# If runtime becomes too long, trim CUTOFFS to [2, 4, 6, 8, 10]

print("=== cutoff vs basis size grid ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Gaussian expansion: {GAUSSIANS} functions")
print(f"Grid: functions_per_center={FUNCTIONS_PER_CENTER_LIST}, cutoff={CUTOFFS}")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

header = (
    f"{'fpc':>3s}  {'cutoff':>6s}  {'E1_err':>16s}  {'abs_E1_err':>16s}  "
    f"{'E2_err':>16s}  {'abs_E2_err':>16s}  "
    f"{'max_H_err':>16s}  {'max_S_err':>16s}  {'t_assem(s)':>10s}"
)
print(header)
print("-" * len(header))

for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis = DisplacedHoBasis(
        center_distance=A,
        functions_per_center=fpc,
        omega=OMEGA,
    )

    real_assembler = RealSpaceMatrixAssembler(quadrature)
    real_h, real_s = real_assembler.assemble(problem, basis)
    real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

    for cutoff in CUTOFFS:
        trunc_assembler = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=cutoff
        )

        t0 = time.perf_counter()
        trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
        t1 = time.perf_counter()
        t_assem = t1 - t0

        trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

        e1_err = trunc_e[0] - real_e[0]
        abs_e1_err = abs(e1_err)

        n_ev = min(len(trunc_e), len(real_e))
        if n_ev > 1:
            e2_err = trunc_e[1] - real_e[1]
            abs_e2_err = abs(e2_err)
        else:
            e2_err = float("nan")
            abs_e2_err = float("nan")

        max_h_err = float(np.max(np.abs(trunc_h - real_h)))
        max_s_err = float(np.max(np.abs(trunc_s - real_s)))

        print(
            f"{fpc:>3d}  {cutoff:>6d}  {e1_err:>+16.10e}  {abs_e1_err:>16.10e}  "
            f"{e2_err:>+16.10e}  {abs_e2_err:>16.10e}  "
            f"{max_h_err:>16.10e}  {max_s_err:>16.10e}  {t_assem:>10.4f}"
        )

print("")
print("=== Interpretation ===")
print("- max_S_err should remain near machine precision (~1e-16): overlap is exact.")
print("- H and eigenvalue errors should generally decrease with cutoff.")
print("- Signed eigenvalue errors may oscillate with cutoff (zero crossings).")
print("- Increasing basis size may increase the required cutoff because more")
print("  high-index (long-range) matrix elements are sampled and need truncation.")
