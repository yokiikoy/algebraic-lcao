from __future__ import annotations

import math

import numpy as np


def associated_laguerre(n: int, alpha: int, x: float) -> float:
    if n < 0:
        raise ValueError("n must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    total = 0.0
    for j in range(n + 1):
        total += (
            (-1.0) ** j
            * math.comb(n + alpha, n - j)
            * x**j
            / math.factorial(j)
        )
    return total


def displacement_matrix_element(n: int, m: int, beta: float) -> float:
    """Return <n|D(beta)|m> for real D(beta)=exp(beta a^dagger - beta a)."""
    if n < 0 or m < 0:
        return 0.0

    beta = float(beta)
    prefactor = math.exp(-0.5 * beta**2)
    if n >= m:
        return (
            prefactor
            * math.sqrt(math.factorial(m) / math.factorial(n))
            * beta ** (n - m)
            * associated_laguerre(m, n - m, beta**2)
        )

    return (
        prefactor
        * math.sqrt(math.factorial(n) / math.factorial(m))
        * (-beta) ** (m - n)
        * associated_laguerre(n, m - n, beta**2)
    )


def displaced_ho_overlap(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
) -> float:
    beta = math.sqrt(omega / 2.0) * (center_right - center_left)
    return displacement_matrix_element(n, m, beta)


def hermite_shifted_coefficients(n: int, center: float, omega: float) -> np.ndarray:
    """Coefficients of H_n(sqrt(omega) * (x - center)) in powers of x."""
    coeffs = np.zeros(n + 1, dtype=float)
    root_omega = math.sqrt(omega)
    for r in range(n // 2 + 1):
        power = n - 2 * r
        term_coeff = (
            (-1.0) ** r
            * math.factorial(n)
            / (math.factorial(r) * math.factorial(power))
            * (2.0 * root_omega) ** power
        )
        for k in range(power + 1):
            coeffs[k] += term_coeff * math.comb(power, k) * (-center) ** (power - k)
    return coeffs


def shifted_gaussian_moment(power: int, exponent: float, center: float) -> float:
    """Return integral x^power exp(-exponent * (x - center)^2) dx on R.

    Expanding x^p = (u + center)^p leaves only even central moments. For
    j=2r, integral u^(2r) exp(-a u^2) du = Gamma(r+1/2)/a^(r+1/2), so the
    sqrt(pi) factor is included as Gamma(1/2) in the r=0 case.
    """
    if exponent <= 0.0:
        raise ValueError("Gaussian exponent must be positive.")

    total = 0.0
    for j in range(power + 1):
        if j % 2 == 1:
            continue
        r = j // 2
        central = math.gamma(r + 0.5) / exponent ** (r + 0.5)
        total += math.comb(power, j) * center ** (power - j) * central
    return total


def ho_norm(n: int, omega: float) -> float:
    return (omega / math.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))


def ho_gaussian_matrix_element(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
    gaussian_exponent: float,
    gaussian_center: float,
) -> float:
    """Return <n,A|exp(-alpha (x-C)^2)|m,B> by finite algebraic sums."""
    if n < 0 or m < 0:
        return 0.0
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    if gaussian_exponent < 0.0:
        raise ValueError("Gaussian exponent must be non-negative.")

    lam = omega + gaussian_exponent
    rho = omega * (center_left + center_right) + 2.0 * gaussian_exponent * gaussian_center
    const = (
        0.5 * omega * (center_left**2 + center_right**2)
        + gaussian_exponent * gaussian_center**2
    )
    shifted_center = rho / (2.0 * lam)
    exponential_prefactor = math.exp(-const + rho**2 / (4.0 * lam))

    left_coeffs = hermite_shifted_coefficients(n, center_left, omega)
    right_coeffs = hermite_shifted_coefficients(m, center_right, omega)
    product_coeffs = np.polynomial.polynomial.polymul(left_coeffs, right_coeffs)

    moment_sum = 0.0
    for power, coeff in enumerate(product_coeffs):
        moment_sum += coeff * shifted_gaussian_moment(power, lam, shifted_center)

    return ho_norm(n, omega) * ho_norm(m, omega) * exponential_prefactor * moment_sum


def kinetic_matrix_element(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
) -> float:
    """Return <n,A|-1/2 d^2/dx^2|m,B> in the displaced HO basis."""
    diagonal = (2 * m + 1) * displaced_ho_overlap(
        n, center_left, m, center_right, omega
    )
    lower = math.sqrt(m * (m - 1)) * displaced_ho_overlap(
        n, center_left, m - 2, center_right, omega
    )
    upper = math.sqrt((m + 1) * (m + 2)) * displaced_ho_overlap(
        n, center_left, m + 2, center_right, omega
    )
    return omega * (diagonal - lower - upper) / 4.0
