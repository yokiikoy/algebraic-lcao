# qmarg

Research prototype for one-dimensional two-center soft-Coulomb LCAO
basis comparison with an algebraic truncated Hamiltonian backend.

## Backend

- **real_space**: exact quadrature-based H and S (reference)
- **algebraic_truncated**: exact overlap S, cutoff-dependent H via
  truncated Gaussian expansions. Eigenvalues converge oscillatorily
  with increasing cutoff.

## Quick start

All experiments are manual diagnostic scripts (no CLI, no CI):

```bash
# Cutoff convergence at fixed basis size
python3 results/run_truncated_backend_convergence.py

# Cutoff x basis-size grid
python3 results/run_cutoff_basis_grid.py

# Cutoff x basis-size extended sweep (fpc 1..8)
python3 results/run_truncated_backend_2d_sweep.py

# Cutoff selection rule table
python3 results/run_cutoff_selection_summary.py

# Stability classification (stable / oscillatory / unstable)
python3 results/run_stability_classification.py

# Basis error vs cutoff error decomposition
python3 results/run_error_decomposition.py

# Solver instability diagnostics
python3 results/run_backend_diagnostics.py

# Odd/even cutoff parity check
python3 results/run_cutoff_odd_even_check.py
```

## Key findings

### 1. S is exact
Overlap matrix error is ~2e-16 across all basis sizes and cutoffs.

### 2. Oscillatory eigenvalue convergence
Eigenvalue errors change sign between successive cutoffs
(e.g., E1_err: + -> - -> + -> -). This is genuine convergence
behaviour, not a parity artifact. Use absolute error and the
last-two-cutoff delta to characterize convergence robustly.

### 3. Cutoff selection rule

| Tolerance | Minimum cutoff | Reachable? |
|-----------|---------------|------------|
| loose (1e-3) | K ≥ 10–12 | Yes |
| medium (1e-4) | K ≥ 16 (estimated) | No (K ≤ 12) |
| strict (1e-5) | K ≫ 12 | No (basis-error limited) |

Cutoff requirement is approximately independent of basis size
for fpc ≤ 4.

### 4. Error decomposition
At K ≥ 10, cutoff error ≪ basis error for fpc ≥ 2.
E2 basis error ~7e-3 dominates — larger basis needed for ε < 1e-3.

### 5. Backend instability at large basis (fpc ≥ 6)
Truncation occasionally flips the sign of near-zero H eigenvalues
(λ ~ 10⁻⁷). The S⁻¹/² transformation amplifies the sign-flipped mode,
producing spurious generalized eigenvalues (E1_err up to 10⁵).
fpc ≤ 4 is always stable.

## Planned Prolog integration

Prolog is intended as a symbolic formula generator and rule checker;
Python remains responsible for numerical evaluation, matrix assembly,
eigensolvers, and convergence experiments. See
[`docs/prolog_plan.md`](docs/prolog_plan.md).

## SU(1,1) Gaussian-operator roadmap

Fixes the non-unitary Gaussian multiplication operator convention
before adding a second algebraic backend. See
[`docs/su11_gaussian_operator.md`](docs/su11_gaussian_operator.md).
