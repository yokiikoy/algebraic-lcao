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
<n,A|H|m,B>
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
|n,c> = T(c)|n>,        <x|n,c> = psi_n^omega(x - c).
```

The real displacement parameter between centers is

```text
beta(A,B) = sqrt(omega / 2) * (B - A)
<n,A|m,B> = <n|D(beta(A,B))|m>.
```

The kinetic operator is

```text
T = p^2 / 2
  = omega/4 * (2 adag a + 1 - a^2 - adag^2).
```

Therefore

```text
T|m> = omega/4 * (
    (2m + 1)|m>
    - sqrt(m(m-1)) |m-2>
    - sqrt((m+1)(m+2)) |m+2>
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
<n|G_alpha|m>.
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

## General Displaced Target

For centers `A`, `B`, and Gaussian center `C`,

```text
<n,A|exp(-alpha (x-C)^2)|m,B>
```

can be reduced to centered Gaussian operators plus real displacements. This
will be introduced only after the centered target has an independent SU(1,1)
backend and tests.

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
