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
* Displaced oscillator basis state: `|n, c⟩ = D(c) |n⟩`, where
  `D(c) = exp(c a† - c a)` is the Heisenberg-Weyl displacement operator
  and `c` is a real displacement parameter.
* Relative displacement between centers `A` and `B`:
  `beta(A, B) = sqrt(omega / 2) * (B - A)`
  such that `bra(n, A) ket(m, B) = bra(n) D(beta(A, B)) ket(m)`.

**Decomposition:**

The displaced Gaussian matrix element is **not** a "pure SU(1,1)" quantity.
It factorizes into:

1. **Displacement** (Heisenberg-Weyl side): the states `|n,A⟩` and `|m,B⟩`
   bring in the displacement operator `D(beta(A,B))`, evaluated via harmonic
   oscillator transformation formulas.
2. **Centered Gaussian operator** (SU(1,1)-consistent side): after state
   displacement, the operator becomes a centered Gaussian
   `exp(-alpha (x - C)^2)` expressed in the origin-centered oscillator basis.
   This part is handled entirely by the existing centered Gaussian APIs
   `origin_gaussian_matrix_element(...)` or `origin_gaussian_matrix_element_su11(...)`.

The full reduction formula is:

```
⟨n,A| exp(-alpha (x-C)^2) |m,B⟩ =
    ⟨n| D(beta(A,B))  exp(-alpha (x - C)^2)  D(beta(A,B))† |m⟩
```

The inner operator `D(beta) exp(-alpha (x-C)^2) D(beta)†` is a Gaussian centered at
`(shifted position)` which then reduces to a centered Gaussian matrix element
once coordinates are appropriately translated. The centered Gaussian matrix
element is evaluated by the existing centered backends; the displacement algebra
is handled separately via Heisenberg-Weyl transformations.

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
