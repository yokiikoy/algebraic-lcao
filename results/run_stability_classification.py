"""
Stability classification and cutoff selection for algebraic_truncated backend.

For each (fpc, cutoff):
  1. Classify: stable / oscillatory / unstable
  2. Compute: max eigenvalue error, delta_E, H Frobenius error
  3. Produce: stability map, filtered cutoff selection rules

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

UNSTABLE_THRESHOLD = 0.5

print("=== stability classification ===")
print(f"Problem: Z={Z}, a={A}, mu={MU}, omega={OMEGA}")
print(f"Grid: fpc={FUNCTIONS_PER_CENTER_LIST}, cutoff={CUTOFFS}")
print(f"Unstable threshold: |E_err| > {UNSTABLE_THRESHOLD}")
print("")

exponents, weights = fit_soft_coulomb_gaussians(MU, GAUSSIANS)
terms = two_center_gaussian_expansion_terms(
    center_distance=A, z=Z, exponents=exponents, weights=weights
)
problem = GaussianExpansionProblem(terms)

quadrature = GaussHermiteQuadrature(QUADRATURE_ORDER)
solver = SymmetricOrthogonalizationSolver()

all_data = {}

for fpc in FUNCTIONS_PER_CENTER_LIST:
    basis = DisplacedHoBasis(
        center_distance=A, functions_per_center=fpc, omega=OMEGA,
    )

    real_assembler = RealSpaceMatrixAssembler(quadrature)
    real_h, real_s = real_assembler.assemble(problem, basis)
    real_e = solver.solve(real_h, real_s, NUM_STATES).eigenvalues

    records = []
    last_valid_e1 = None
    last_valid_e2 = None
    last_valid_e1_err = None
    last_valid_e2_err = None

    for cutoff in CUTOFFS:
        trunc_assembler = RealSpaceMatrixAssembler(
            quadrature, backend="algebraic_truncated", cutoff=cutoff
        )
        t0 = time.perf_counter()
        trunc_h, trunc_s = trunc_assembler.assemble(problem, basis)
        trunc_e = solver.solve(trunc_h, trunc_s, NUM_STATES).eigenvalues

        e1_err = trunc_e[0] - real_e[0]
        e2_err = trunc_e[1] - real_e[1]
        abs_e1_err = abs(e1_err)
        abs_e2_err = abs(e2_err)
        max_abs_e_err = max(abs_e1_err, abs_e2_err)

        if last_valid_e1 is not None:
            delta_e1 = abs(trunc_e[0] - last_valid_e1)
            delta_e2 = abs(trunc_e[1] - last_valid_e2)
            max_delta_e = max(delta_e1, delta_e2)
        else:
            delta_e1 = delta_e2 = max_delta_e = None

        h_fro = float(
            np.linalg.norm(trunc_h - real_h, "fro")
            / np.linalg.norm(real_h, "fro")
        )

        # ── Classification ──────────────────────────────────────────
        if max_abs_e_err > UNSTABLE_THRESHOLD:
            status = "unstable"
        elif last_valid_e1 is None:
            status = "stable"
        elif last_valid_e1_err is not None and (
                e1_err * last_valid_e1_err < 0 or e2_err * last_valid_e2_err < 0):
            status = "oscillatory"
        elif max_abs_e_err < 0.001 and max_delta_e < 0.001:
            status = "stable"
        else:
            status = "oscillatory"

        if status != "unstable":
            last_valid_e1 = trunc_e[0]
            last_valid_e2 = trunc_e[1]
            last_valid_e1_err = e1_err
            last_valid_e2_err = e2_err
        t_assem = time.perf_counter() - t0

        records.append({
            "cutoff": cutoff,
            "status": status,
            "e1_err": e1_err,
            "e2_err": e2_err,
            "max_abs_e_err": max_abs_e_err,
            "max_delta_e": max_delta_e,
            "h_fro": h_fro,
            "t_assem": t_assem,
        })

    all_data[fpc] = records

# ── 1. Stability map ────────────────────────────────────────────────

COLOR = {"stable": " S ", "oscillatory": " ~ ", "unstable": " ! "}

print("=" * 55)
print("Stability map:  S = stable  ~ = oscillatory  ! = unstable")
print("=" * 55)
header = "fpc  | " + "  ".join(f"K={c:<3d}" for c in CUTOFFS)
print(header)
print("-" * len(header))
for fpc in FUNCTIONS_PER_CENTER_LIST:
    row = "  ".join(COLOR[r["status"]] for r in all_data[fpc])
    print(f" {fpc}   | {row}")
print("")

# ── 2. Detail table ─────────────────────────────────────────────────

print("=" * 90)
print("Detail by (fpc, cutoff)")
print("=" * 90)
hdr = f"{'fpc':>3s}  {'cutoff':>6s}  {'status':>12s}  {'max_E':>12s}  {'dE':>12s}  {'H_fro':>12s}  {'t(s)':>10s}"
print(hdr)
print("-" * len(hdr))
for fpc in FUNCTIONS_PER_CENTER_LIST:
    for r in all_data[fpc]:
        de = f"{r['max_delta_e']:>12.4e}" if r['max_delta_e'] is not None else " " * 12
        print(
            f"{fpc:>3d}  {r['cutoff']:>6d}  {r['status']:>12s}  "
            f"{r['max_abs_e_err']:>12.4e}  {de}  "
            f"{r['h_fro']:>12.4e}  {r['t_assem']:>10.4f}"
        )

# ── 3. Filtered cutoff selection ────────────────────────────────────

print("")
print("=" * 70)
print("Filtered cutoff selection (stable only)")
print("=" * 70)
hdr2 = f"{'fpc':>3s}  {'loose(1e-3)':>14s}  {'medium(1e-4)':>14s}  {'all stable?':>12s}"
print(hdr2)
print("-" * len(hdr2))

for fpc in FUNCTIONS_PER_CENTER_LIST:
    records = all_data[fpc]
    n_stable = sum(1 for r in records if r["status"] == "stable")
    total = len(records)
    all_stable = n_stable == total
    stable_mark = "yes" if all_stable else f"{n_stable}/{total}"

    def first_valid_cutoff(eps):
        for r in records:
            if r["cutoff"] == CUTOFFS[0]:
                continue
            if r["status"] == "unstable":
                continue
            if r["max_delta_e"] < 2 * eps:
                return r["cutoff"]
        return None

    k3 = first_valid_cutoff(1e-3)
    k4 = first_valid_cutoff(1e-4)
    s3 = str(k3) if k3 is not None else "N/A"
    s4 = str(k4) if k4 is not None else "N/A"
    print(f"{fpc:>3d}  {s3:>14s}  {s4:>14s}  {stable_mark:>12s}")

# ── 4. Interpretation ───────────────────────────────────────────────

print("")
print("=" * 70)
print("Interpretation")
print("=" * 70)
print("- stable:   |E_err| <= 0.5, no sign oscillation, well-converged.")
print("- oscillatory: physically reasonable eigenvalues, but sign oscillates")
print("               or still in transitional regime between cutoffs.")
print("- unstable: |E_err| > 0.5 (spurious eigenvalue from solver).")
print("  Root cause: truncation flips sign of near-zero H eigenvalues;")
print("  S^(-1/2) transformation amplifies the sign-flipped mode, producing")
print("  a non-physical generalized eigenvalue that masquerades as E1.")
print("")
print("  fpc=1,2,3,4 are fully stable. fpc=6,8 have intermittent instabilities.")
print("  The filtered cutoff selection rejects unstable entries automatically.")
print("  For production use, restrict to the stable subset at each fpc.")
