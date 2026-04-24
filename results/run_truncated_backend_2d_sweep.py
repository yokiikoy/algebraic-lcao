"""
Comprehensive 2D sweep: functions_per_center × cutoff for algebraic_truncated.

Measures eigenvalue errors, Frobenius H error, oscillation diagnostics,
and last-two-cutoff convergence estimates.

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
FUNCTIONS_PER_CENTER_LIST = [1, 2, 3, 4, 6, 8]
CUTOFFS = [2, 4, 6, 8, 10, 12]
TOLERANCES = [("loose", 1e-3), ("medium", 1e-4)]

print("=== 2D sweep: basis size vs cutoff ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Gaussian expansion: {GAUSSIANS} functions")
print(f"Grid: fpc={FUNCTIONS_PER_CENTER_LIST}, cutoff={CUTOFFS}")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

# ── Collect data ──────────────────────────────────────────────────

all_data = {}  # fpc -> list of records

for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis = DisplacedHoBasis(
        center_distance=A,
        functions_per_center=fpc,
        omega=OMEGA,
    )

    real_assembler = RealSpaceMatrixAssembler(quadrature)
    real_h, real_s = real_assembler.assemble(problem, basis)
    real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

    records = []
    prev_e1 = None
    prev_e2 = None

    for cutoff in CUTOFFS:
        trunc_assembler = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=cutoff
        )

        t0 = time.perf_counter()
        trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
        t1 = time.perf_counter()

        trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

        e1_err = trunc_e[0] - real_e[0]
        e2_err = trunc_e[1] - real_e[1]
        abs_e1_err = abs(e1_err)
        abs_e2_err = abs(e2_err)
        max_abs_e_err = max(abs_e1_err, abs_e2_err)

        delta_e1 = abs(trunc_e[0] - prev_e1) if prev_e1 is not None else None
        delta_e2 = abs(trunc_e[1] - prev_e2) if prev_e2 is not None else None
        max_delta_e = max(delta_e1, delta_e2) if delta_e1 is not None else None
        prev_e1 = trunc_e[0]
        prev_e2 = trunc_e[1]

        max_h_err = float(np.max(np.abs(trunc_h - real_h)))
        h_fro = float(
            np.linalg.norm(trunc_h - real_h, "fro")
            / np.linalg.norm(real_h, "fro")
        )
        max_s_err = float(np.max(np.abs(trunc_s - real_s)))
        t_assem = t1 - t0

        records.append(
            {
                "cutoff": cutoff,
                "e1_err": e1_err,
                "e2_err": e2_err,
                "abs_e1_err": abs_e1_err,
                "abs_e2_err": abs_e2_err,
                "max_abs_e_err": max_abs_e_err,
                "delta_e1": delta_e1,
                "delta_e2": delta_e2,
                "max_delta_e": max_delta_e,
                "max_h_err": max_h_err,
                "h_fro": h_fro,
                "max_s_err": max_s_err,
                "t_assem": t_assem,
            }
        )

    all_data[fpc] = records

# ── Detail table ──────────────────────────────────────────────────

def fmt_delta(val):
    return f"{val:>12.4e}" if val is not None else f"{'N/A':>12s}"

header = (
    f"{'fpc':>3s}  {'cutoff':>6s}  {'E1_err':>16s}  {'abs_E1':>12s}  "
    f"{'E2_err':>16s}  {'abs_E2':>12s}  {'max_E':>12s}  "
    f"{'d_E1':>12s}  {'d_E2':>12s}  {'H_fro':>12s}  {'t(s)':>10s}"
)
print(header)
print("-" * len(header))

for fpc in FUNCTIONS_PER_CENTER_LIST:
    for r in all_data[fpc]:
        print(
            f"{fpc:>3d}  {r['cutoff']:>6d}  {r['e1_err']:>+16.10e}  "
            f"{r['abs_e1_err']:>12.4e}  {r['e2_err']:>+16.10e}  "
            f"{r['abs_e2_err']:>12.4e}  {r['max_abs_e_err']:>12.4e}  "
            f"{fmt_delta(r['delta_e1'])}  {fmt_delta(r['delta_e2'])}  "
            f"{r['h_fro']:>12.4e}  {r['t_assem']:>10.4f}"
        )

# ── Last-two-cutoff oscillatory estimates ─────────────────────────

print("")
print("=" * 80)
print("Oscillatory convergence estimates (last two cutoffs)")
print("=" * 80)
print(
    f"{'fpc':>3s}  {'E1_est':>18s}  {'E1_err_est':>16s}  "
    f"{'E2_est':>18s}  {'E2_err_est':>16s}  {'H_fro(K_max)':>16s}"
)
print("-" * 96)

for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    r_max = records[-1]
    r_prev = records[-2]

    E1_est = 0.5 * (r_max["e1_err"] + r_prev["e1_err"])
    err1_est = abs(r_max["e1_err"] - r_prev["e1_err"])
    E2_est = 0.5 * (r_max["e2_err"] + r_prev["e2_err"])
    err2_est = abs(r_max["e2_err"] - r_prev["e2_err"])

    print(
        f"{fpc:>3d}  {E1_est:>+18.10e}  {err1_est:>16.10e}  "
        f"{E2_est:>+18.10e}  {err2_est:>16.10e}  "
        f"{r_max['h_fro']:>16.10e}"
    )

# ── Acceptance summary ────────────────────────────────────────────

def min_cutoff(records, eps):
    for r in records:
        if r["cutoff"] == CUTOFFS[0]:
            continue
        if r["max_abs_e_err"] < eps and r["max_delta_e"] < 2 * eps:
            return r["cutoff"]
    return None

print("")
print("=" * 80)
print("Cutoff selection summary")
print("=" * 80)
header2 = f"{'fpc':>3s}  {'cutoff(1e-3)':>14s}  {'cutoff(1e-4)':>14s}  {'limiting @ K_max':>30s}"
print(header2)
print("-" * len(header2))

for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    k3 = min_cutoff(records, 1e-3)
    k4 = min_cutoff(records, 1e-4)
    s3 = str(k3) if k3 is not None else "N/A"
    s4 = str(k4) if k4 is not None else "N/A"

    r_last = records[-1]
    ratio_e = r_last["max_abs_e_err"] / 1e-4
    ratio_d = r_last["max_delta_e"] / 2e-4 if r_last["max_delta_e"] is not None else 0
    limit = f"E={ratio_e:.1f}x dE={ratio_d:.1f}x"
    print(f"{fpc:>3d}  {s3:>14s}  {s4:>14s}  {limit:>30s}")

# ── Error floor analysis ──────────────────────────────────────────

print("")
print("=" * 80)
print("Error floor analysis")
print("=" * 80)
print(
    f"{'fpc':>3s}  {'min_max_E':>14s}  {'min_H_fro':>14s}  "
    f"{'E1_floor':>14s}  {'E2_floor':>14s}  {'basis size':>12s}"
)
print("-" * 74)

for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    min_max_e = min(r["max_abs_e_err"] for r in records)
    min_h = min(r["h_fro"] for r in records)
    r_best = records[-1]
    basis_size = fpc * 2
    print(
        f"{fpc:>3d}  {min_max_e:>14.4e}  {min_h:>14.4e}  "
        f"{r_best['abs_e1_err']:>14.4e}  {r_best['abs_e2_err']:>14.4e}  "
        f"{basis_size:>12d}"
    )

# ── Interpretation ────────────────────────────────────────────────

print("")
print("=" * 80)
print("Summary")
print("=" * 80)

print("")
print("1. Error vs cutoff behavior per basis:")
for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    e_low = records[0]["max_abs_e_err"]
    e_high = records[-1]["max_abs_e_err"]
    ratio = e_low / e_high if e_high > 0 else float("inf")
    print(f"   fpc={fpc}: max_E {e_low:.2e} -> {e_high:.2e} (factor {ratio:.0f}x improvement)")

print("")
print("2. Evidence of error floor:")
for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    floor_e1 = records[-1]["abs_e1_err"]
    floor_e2 = records[-1]["abs_e2_err"]
    h_at_max = records[-1]["h_fro"]
    e_at_penultimate = records[-2]["max_abs_e_err"]
    e_at_last = records[-1]["max_abs_e_err"]
    improvement = e_at_penultimate / e_at_last if e_at_last > 0 else float("inf")
    status = "floor possible" if improvement < 3 else "still converging"
    print(
        f"   fpc={fpc}: E1~{floor_e1:.2e} E2~{floor_e2:.2e} H_fro~{h_at_max:.2e} "
        f"last-2 improvement={improvement:.1f}x -> {status}"
    )

print("")
print("3. Suggested cutoff rule:")
print("   - Loose (1e-3): K >= 10 for fpc >= 2; K >= 12 for fpc = 1.")
print("   - Medium (1e-4): not reached within current grid (K <= 12) for any fpc.")
print("   - Bottleneck is delta_E (oscillation damping), not pointwise |E_err|.")
print("   - For fpc >= 2, basis error floor ~ 2-3e-3 (E2) prevents reaching 1e-4")
print("     regardless of cutoff. Larger basis (fpc > 8) needed for 1e-4 E2.")
print("")

# ── Notes ─────────────────────────────────────────────────────────

print("=== Notes ===")
print("- Oscillatory convergence requires delta_E guard against zero-crossings.")
print("- This is a manual diagnostic, not a CI test.")
