# SU(1,1) Gaussian Operator Roadmap

This note fixes the operator convention before adding a second algebraic
backend for Gaussian multiplication operators.

## Goal

For a one-dimensional Gaussian-expanded Hamiltonian,

```text
H = p^2 / 2 + sum_k w_k exp(-alpha_k (x - R_k)^2),
```

compute displaced-oscillator matrix elements

```text
bra(n,A) H ket(m,B)
```

without numerical quadrature.

The current implementation already does this with finite Hermite-polynomial and
Gaussian-moment sums. The next theoretical backend should express the same
calculation through the Heisenberg-Weyl and SU(1,1) operator algebra.

## Fixed Convention

The normalized oscillator basis uses width `omega`:

```text
x = (a + adag) / sqrt(2 omega)
p = i sqrt(omega / 2) (adag - a)
[a, adag] = 1
```

Displaced states are

```text
ket(n,c) = T(c) ket(n)
coordinate wavefunction: psi_n^omega(x - c)
```

The real displacement parameter between centers is

```text
beta(A,B) = sqrt(omega / 2) * (B - A)
bra(n,A) ket(m,B) = bra(n) D(beta(A,B)) ket(m)
```

The kinetic operator is

```text
T = p^2 / 2
  = omega/4 * (2 adag a + 1 - a^2 - adag^2).
```

Therefore

```text
T ket(m) = omega/4 * (
    (2m + 1) ket(m)
    - sqrt(m(m-1)) ket(m-2)
    - sqrt((m+1)(m+2)) ket(m+2)
).
```

This is the convention implemented by `kinetic_matrix_element`.

## Gaussian Multiplication Operator

Define the centered Gaussian operator

```text
G_alpha = exp(-alpha x^2)
        = exp[-alpha/(2 omega) * (a + adag)^2].
```

Expanded in SU(1,1) generators,

```text
K+ = adag^2 / 2
K- = a^2 / 2
K0 = (adag a + 1/2) / 2

G_alpha = exp[-(alpha / omega) * (K+ + K- + 2 K0)].
```

Important: `G_alpha` is not a unitary squeeze operator. It is a positive
Gaussian multiplication operator, equivalently a non-unitary element in the
complexified SU(1,1) semigroup. Naming it as a plain `squeeze` would hide this
distinction.

Use names such as:

```text
origin_gaussian_matrix_element
su11_gaussian_matrix_element
```

instead of `squeeze_matrix_element` for this project.

## Centered Target

The first SU(1,1) target is only

```text
bra(n) G_alpha ket(m)
```

Expected structural properties:

- it is zero when `n + m` is odd
- it is symmetric in `n,m`
- it agrees with
  `ho_gaussian_matrix_element(n, 0, m, 0, omega, alpha, 0)`

The current code exposes this target as

```text
origin_gaussian_matrix_element(n, m, omega, alpha)
```

and delegates to the Hermite-moment backend until the SU(1,1) normal-form
backend is implemented.

The independent centered comparison backend is

```text
origin_gaussian_matrix_element_su11(n, m, omega, alpha)
```

It currently covers only `bra(n) G_alpha ket(m)` and is used for dual-backend
validation against the Hermite-moment target API. It is a centered Gaussian
finite-sum backend consistent with the SU(1,1) viewpoint, but not yet an
explicit SU(1,1) normal-form implementation.

## General Displaced Target

For centers `A`, `B`, and Gaussian center `C`,

```text
bra(n,A) exp(-alpha (x-C)^2) ket(m,B)
```

can be reduced to centered Gaussian operators plus Heisenberg-Weyl displacements.
The reduction proceeds by expressing the displaced states in terms of the
origin-centered harmonic-oscillator basis and then collecting the centered
Gaussian operator.

**Conventions:**

* Ladder-operator algebra: `[a, adag] = 1`
* Coordinate operator: `x = (a + adag) / sqrt(2 * omega)`
* Displaced oscillator basis state: `|n, c⟩ = D(β_c) |n⟩`,
  with `β_c = sqrt(omega / 2) * c`.
* Displacement operator `D(β) = exp(β a† - β a)` (Heisenberg-Weyl).

**Decomposition:**

The displaced Gaussian matrix element is **not** a "pure SU(1,1)" quantity.
It factorizes into:

1. **Displacement** (Heisenberg-Weyl side): the displaced bra and ket
   introduce separate displacement operators: `⟨n,A| = ⟨n| D(-β_A)`,
   `|m,B⟩ = D(β_B)|m⟩`, where `β_A = sqrt(omega / 2) * A`,
   `β_B = sqrt(omega / 2) * B`.
2. **Centered Gaussian operator** (SU(1,1)-consistent side):
   The Gaussian `G_{α,C} = exp(-α (x - C)^2)` can be written as
   `G_{α,C} = D(β_C) G_{α,0} D(-β_C)` with
   `β_C = sqrt(omega / 2) * C`.

Thus the full decomposition separates the Heisenberg-Weyl (displacement)
from the centered Gaussian operator (SU(1,1)-consistent):

```
⟨n,A| G_{α,C} |m,B⟩
===================================

    = ⟨n| D(-β_A) · D(β_C) G_{α,0} D(-β_C) · D(β_B) |m⟩
    = ⟨n| D(β_C - β_A)  G_{α,0}  D(β_B - β_C) |m⟩
```

Here:

* `G_{α,0} = exp(-α x^2)` is the **centered Gaussian operator**,
  which is handled entirely by the existing centered Gaussian APIs
  `origin_gaussian_matrix_element(...)` or `origin_gaussian_matrix_element_su11(...)`.
* The remaining `D(β_C - β_A)` and `D(β_B - β_C)` are **Heisenberg-Weyl
  displacement operators** that must be evaluated via harmonic-oscillator
  transformation formulas.

This decomposition clearly shows the strategy for future implementation:
first compute the centered Gaussian matrix element using the existing
centered backend, then apply the necessary displacement bookkeeping.

**Implementation Roadmap:**

* Centered dual-backend validation (`origin_gaussian_matrix_element_su11`) is
  complete and tested.
* The next step is a small helper for displaced reduction: a function that,
  given `(n, m, omega, alpha, A, B, C)`, computes the displacement parameters
  and delegates the centered Gaussian matrix element to the existing centered
  backend.
* Only after the displaced reduction helper is in place should a full displaced
  backend implementation be attempted.

## Implementation Order

1. Keep the existing Hermite-moment backend as the reference algebraic backend.
2. Add the centered `origin_gaussian_matrix_element` API and tests.
3. Implement an independent `su11_gaussian_matrix_element` for the centered
   operator.
4. Compare the SU(1,1) backend against `origin_gaussian_matrix_element` for
   small `n,m`.
5. Only then extend the SU(1,1) backend to displaced/multi-center matrix
   elements.
6. Keep both backends until numerical stability and range are understood.

The immediate goal is not to delete the Hermite-moment implementation. It is to
establish a second derivation path that returns the same numbers.

## Evaluation Order and Interface

In operator algebra, operators act on the ket from **right to left**.
For the displaced Gaussian target this means:

1. **First** apply the right displacement `D(β_B - β_C)` to `|m⟩`
2. **Then** apply the centered Gaussian operator `G_{α,0}`
3. **Finally** apply the left displacement `D(β_C - β_A)` and contract with `⟨n|`

This produces the three-factor structure:

```
⟨n,A| G_{α,C} |m,B⟩ = ⟨n| D(β_left) · G_{α,0} · D(β_right) |m⟩
```

with

```
β_left  = β_C - β_A = sqrt(omega/2) * (C - A)
β_right = β_B - β_C = sqrt(omega/2) * (B - C)
```

### Interface between Prolog and Python

The current codebase separates concerns as follows:

* **Prolog** (term generator): produces the displacement finite-sum structure
  for `<n | D(beta) | k>` via `displacement_term/5`. It does NOT generate
  Gaussian operator terms yet.

* **Python** (numerical evaluator): provides the centered Gaussian operator
  backends `origin_gaussian_matrix_element(...)` and
  `origin_gaussian_matrix_element_su11(...)`.

* **Composition** (explicit factorization): a small helper
  `displaced_gaussian_factorization(center_left, center_right, gaussian_center, omega)`
  returns the `(β_left, β_right)` pair needed by the decomposition above.
  This makes the interface concrete without replacing any backend.

A future full implementation will assemble:

```
⟨n| D(β_left) |k⟩   (Prolog-generated or Python closed-form)
⟨k| G_{α,0} |l⟩     (existing centered Python backend)
⟨l| D(β_right) |m⟩  (Prolog-generated or Python closed-form)
```

and contract over intermediate indices `k, l`.

For now, Prolog generates only the displacement side; the Gaussian-operator
side remains in Python.
