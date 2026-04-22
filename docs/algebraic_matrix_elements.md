# Algebraic Matrix Elements

This project uses the normalized one-dimensional oscillator convention

```text
psi_n^omega(x) = N_n H_n(sqrt(omega) x) exp(-omega x^2 / 2)
```

and displaced basis functions

```text
|n,c> = T(c)|n>,        <x|n,c> = psi_n^omega(x - c).
```

The real displacement parameter in Fock notation is

```text
beta(A,B) = sqrt(omega / 2) * (B - A)
```

so the overlap is

```text
<n,A|m,B> = <n|D(beta(A,B))|m>.
```

For real `beta`,

```text
<n|D(beta)|m>
  = exp(-beta^2/2) sqrt(m!/n!) beta^(n-m) L_m^(n-m)(beta^2),      n >= m
  = exp(-beta^2/2) sqrt(n!/m!) (-beta)^(m-n) L_n^(m-n)(beta^2),   m > n.
```

The kinetic energy is translation invariant. Acting on the ket side,

```text
-1/2 d^2/dx^2 |m,B>
  = omega/4 * ((2m+1)|m,B>
      - sqrt(m(m-1)) |m-2,B>
      - sqrt((m+1)(m+2)) |m+2,B>).
```

For a Gaussian potential term

```text
c exp(-alpha (x - C)^2),
```

the required matrix element is

```text
<n,A|exp(-alpha (x-C)^2)|m,B>.
```

This is evaluated without quadrature by expanding the finite Hermite
polynomials and reducing the result to shifted Gaussian moments. The exponent
combines as

```text
-omega/2 (x-A)^2 - omega/2 (x-B)^2 - alpha (x-C)^2
  = -lambda (x-s)^2 - kappa + rho^2/(4 lambda)

lambda = omega + alpha
rho    = omega(A+B) + 2 alpha C
s      = rho / (2 lambda)
kappa  = omega(A^2+B^2)/2 + alpha C^2.
```

The remaining finite terms use

```text
int_R x^p exp(-lambda (x-s)^2) dx
  = sum_even_j binom(p,j) s^(p-j) Gamma(j/2 + 1/2) / lambda^(j/2 + 1/2).
```

This is the first algebraic assembler target. It deliberately avoids numerical
quadrature while staying directly comparable to the real-space assembler.

`exp(-alpha x^2)` is a positive Gaussian multiplication operator, not a unitary
squeeze by itself. It can be related to squeezing/Bogoliubov algebra, but the
implementation uses the finite Hermite-Gaussian moment formula above because it
is explicit, stable for small basis sizes, and easy to verify against the
existing real-space assembler.
