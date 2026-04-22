# Prolog Integration Plan

This project will use Prolog as a symbolic rule engine, not as the numerical
linear algebra backend.

The current Python implementation already evaluates the required Fock/HO matrix
elements numerically from closed finite formulas:

- `project/qmarg/fock.py`
- `project/qmarg/algebraic_assembler.py`
- `docs/algebraic_matrix_elements.md`

The Prolog work should preserve that boundary. Prolog generates or checks
operator-algebra formulas; Python evaluates those formulas, assembles matrices,
solves generalized eigenproblems, and runs convergence experiments.

## Role Split

Prolog responsibilities:

- represent operator expressions as terms
- encode ladder-operator rewrite rules
- derive matrix-element finite sums
- extract selection rules
- emit simple machine-readable expression terms

Python responsibilities:

- evaluate generated formulas with floating-point arithmetic
- construct `H` and `S`
- solve `H c = E S c`
- compare against real-space and finite-difference references
- run optimization and convergence experiments

Do not move numerical eigensolvers or large matrix evaluation into Prolog.

## Fixed Operator Convention

Use the same real Fock convention as the Python code:

```text
[a, adag] = 1
|n> = (adag)^n / sqrt(n!) |0>
a |n> = sqrt(n) |n-1>
adag |n> = sqrt(n+1) |n+1>
```

For displaced harmonic-oscillator states,

```text
|n,c> = T(c)|n>
beta(A,B) = sqrt(omega / 2) * (B - A)
<n,A|m,B> = <n|D(beta(A,B))|m>
```

The first Prolog target is the real displacement operator

```text
D(beta) = exp(beta adag - beta a)
        = exp(-beta^2/2) exp(beta adag) exp(-beta a)
```

## Milestones

### P0: Prolog skeleton

Add `priv/prolog/` with a small, dependency-free SWI-Prolog program and Python
tests that can call it through `subprocess`.

Acceptance criteria:

- `swipl -q -s priv/prolog/main.pl -g "..."` runs in CI/local tests
- Python tests skip cleanly if `swipl` is unavailable
- no numerical matrix assembly is implemented in Prolog

### P1: Primitive ladder matrix element

Implement the closed primitive

```text
<n| (adag)^p a^q |m>
```

as a Prolog relation that returns a symbolic coefficient and a delta condition,
or a canonical finite expression term.

Expected formula:

```text
a^q |m> = sqrt(m! / (m-q)!) |m-q>,           q <= m
(adag)^p |m-q> = sqrt((m-q+p)! / (m-q)!) |m-q+p>

<n| (adag)^p a^q |m>
  = sqrt(m! (m-q+p)!)/(m-q)! * delta(n, m-q+p)
```

Acceptance criteria:

- exact zero when `q > m`
- correct delta target `n = m - q + p`
- Python test compares Prolog output against a direct Python helper for small
  `0 <= n,m,p,q <= 6`

### P2: Displacement finite-sum generator

Use the BCH form of `D(beta)` to generate

```text
<n|D(beta)|m>
  = exp(-beta^2/2)
    sum_{p,q >= 0}
      beta^p/p! * (-beta)^q/q! *
      <n| (adag)^p a^q |m>
```

The relation should emit a finite sum by applying the primitive delta condition
instead of expanding an infinite series blindly.

Acceptance criteria:

- generated expression evaluates to the same value as
  `qmarg.fock.displacement_matrix_element`
- test grid covers `0 <= n,m <= 6` and several real `beta`
- canonical output is stable enough for snapshot-like tests

### P3: Selection rules and formula metadata

Add Prolog rules that expose metadata:

- displacement has no parity selection rule
- `exp(-alpha x^2)` around the origin only couples states with even `n+m`
- kinetic energy in an undisplaced HO basis only couples `m`, `m-2`, and `m+2`

Acceptance criteria:

- Python tests assert expected allowed/forbidden cases
- metadata is used in at least one Python-side diagnostic or report

### P4: Gaussian operator derivation support

Do not start by asking Prolog to numerically evaluate

```text
<n|exp(-alpha x^2)|m>
```

Instead, use Prolog to derive or check structure:

- expansion in powers of `x = (a + adag)/sqrt(2)`
- parity selection
- finite polynomial/moment terms for bounded `n,m`

Python remains the evaluator using `ho_gaussian_matrix_element`.

Acceptance criteria:

- Prolog can emit the parity rule and bounded term structure
- Python tests compare Prolog-derived structure with Python matrix elements
  for small `n,m`

### P5: Algebraic assembler provenance

Once P1-P4 are stable, wire formula provenance into the algebraic assembler:

- record whether overlap came from displacement formula
- record whether Gaussian potential came from Hermite-moment formula
- optionally print Prolog-derived expression IDs in diagnostics

Acceptance criteria:

- `python3 project/cli/run.py algebraic-check` still matches real-space
  assembly to machine precision
- provenance is visible without changing numerical results

## Non-Goals

- no Prolog eigensolver
- no large floating-point matrix construction in Prolog
- no replacement of NumPy numerical evaluation
- no broad symbolic CAS
- no immediate multi-dimensional generalization

## Recommended Next Task

Start with P0 and P1 only.

That is the smallest useful Prolog slice:

```text
<n| (adag)^p a^q |m>
```

It tests the operator-algebra representation, validates SWI-Prolog integration,
and gives a firm base for the displacement operator without taking on the full
Gaussian operator at once.
