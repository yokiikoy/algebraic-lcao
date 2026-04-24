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

    Shift `u = x - center`, then expand `x^p = (u + center)^p`.
    Odd powers of `u` integrate to zero over the real line. For the remaining
    even powers `j = 2r`,

        integral u^(2r) exp(-a u^2) du = Gamma(r+1/2) / a^(r+1/2).

    The usual sqrt(pi) factor is therefore included as Gamma(1/2) when r=0.
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
    """Return <n,A|exp(-alpha (x-C)^2)|m,B> by finite algebraic sums.

    The Hermite parts from the left and right oscillator states are expanded as
    polynomials in `x`, multiplied together, and combined with the three
    Gaussian exponentials into one shifted Gaussian. The final value is a
    finite sum of polynomial coefficients times shifted Gaussian moments.
    """
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


def origin_gaussian_matrix_element(
    n: int,
    m: int,
    omega: float,
    gaussian_exponent: float,
) -> float:
    """Return <n|exp(-alpha x^2)|m> in the oscillator basis.

    This is the public centered target API for the future SU(1,1)-based
    derivation path. It names the non-unitary Gaussian multiplication operator
    without implying a unitary squeeze implementation.

    The implementation currently delegates to the Hermite-moment backend; a
    future SU(1,1) normal-form backend must match this function before any
    replacement of the current backend.
    """
    return ho_gaussian_matrix_element(
        n,
        0.0,
        m,
        0.0,
        omega,
        gaussian_exponent,
        0.0,
    )


def displaced_gaussian_factorization(
    center_left: float,
    center_right: float,
    gaussian_center: float,
    omega: float,
) -> tuple[float, float]:
    """Return the displacement parameters for the displaced Gaussian decomposition.

    Given centers A (bra), B (ket), and C (Gaussian), the displaced target
    ⟨n,A|G_{α,C}|m,B⟩ decomposes as:

        ⟨n| D(β_left) · G_{α,0} · D(β_right) |m⟩

    where:
        β_left  = sqrt(omega/2) * (C - A)
        β_right = sqrt(omega/2) * (B - C)

    This helper makes the factorization explicit at the interface level.
    It does NOT evaluate any matrix elements; callers must compose the
    displacement and centered Gaussian backends themselves.

    Returns:
        (beta_left, beta_right)
    """
    scale = math.sqrt(omega / 2.0)
    beta_left = scale * (gaussian_center - center_left)
    beta_right = scale * (center_right - gaussian_center)
    return beta_left, beta_right


def _evaluate_displacement_terms(
    terms: tuple, beta: float
) -> float:
    """Evaluate displacement finite sum from Prolog terms:
    exp(-beta^2/2) * Σ_term beta^p/p! * (-beta)^q/q! * sqrt(source! * target!) / den!.
    """
    import math

    prefactor = math.exp(-0.5 * beta ** 2)
    total = 0.0
    for term in terms:
        total += (
            term.evaluate_beta_factor(beta) * term.ladder_coefficient.evaluate()
        )
    return prefactor * total


def displaced_gaussian_matrix_element_truncated_prolog(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
    alpha: float,
    gaussian_center: float,
    cutoff: int = 12,
) -> float:
    """
    Return ⟨n,A|exp(-α (x-C)²)|m,B⟩ via explicit triple sum over intermediate
    oscillator states.

    This is a **truncated** expansion intended as a validation/exploratory
    backend, not an exact closed-form evaluator. The `cutoff` parameter controls
    the truncation of intermediate states: the upper bound for the intermediate
    indices i and j is M = max(n, m) + cutoff (inclusive). Larger `cutoff` values
    should improve accuracy at increased computational cost. This backend is
    designed for validation and exploration, not for final exact assembly.

    Decomposition:
        ⟨n,A|G_{α,C}|m,B⟩ = ⟨n| D(β_L) G_{α,0} D(β_R) |m⟩
    where β_L = √(ω/2) (C - A), β_R = √(ω/2) (B - C).

    Expansion:
        Σ_{i=0}^{M} Σ_{j=0}^{M}
            ⟨n|D(β_L)|i⟩ ⟨i|G_{α,0}|j⟩ ⟨j|D(β_R)|m⟩
    with M = max(n, m) + cutoff (inclusive upper bound).

    Each factor is evaluated via Prolog-generated algebraic terms:
      - Displacement terms from query_displacement_terms
      - Gaussian terms from query_gaussian_terms
    """
    from qmarg.prolog_bridge import (
        evaluate_gaussian_terms,
        query_displacement_terms,
        query_gaussian_terms,
    )

    beta_left, beta_right = displaced_gaussian_factorization(
        center_left, center_right, gaussian_center, omega
    )

    max_index = max(n, m) + cutoff

    displacement_cache: dict[tuple[int, int], tuple] = {}
    gaussian_cache: dict[tuple[int, int], tuple] = {}

    total = 0.0
    for i in range(max_index + 1):
        key_left = (n, i)
        if key_left not in displacement_cache:
            displacement_cache[key_left] = query_displacement_terms(n, i)
        left_terms = displacement_cache[key_left]
        left_val = _evaluate_displacement_terms(left_terms, beta_left)
        if left_val == 0.0:
            continue
        for j in range(max_index + 1):
            key_gauss = (i, j)
            if key_gauss not in gaussian_cache:
                gaussian_cache[key_gauss] = query_gaussian_terms(i, j)
            gauss_terms = gaussian_cache[key_gauss]
            gauss_val = evaluate_gaussian_terms(gauss_terms, omega, alpha)
            if gauss_val == 0.0:
                continue
            key_right = (j, m)
            if key_right not in displacement_cache:
                displacement_cache[key_right] = query_displacement_terms(j, m)
            right_terms = displacement_cache[key_right]
            right_val = _evaluate_displacement_terms(right_terms, beta_right)
            total += left_val * gauss_val * right_val

    return total


def displaced_gaussian_matrix_element_truncated(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
    alpha: float,
    gaussian_center: float,
    cutoff: int | None = None,
) -> float:
    """
    Return ⟨n,A|exp(-α (x-C)²)|m,B⟩ using a truncated intermediate-state expansion.
    
    This is a **validation backend** (approximate, not exact) that depends on
    the `cutoff` parameter. The `cutoff` controls the truncation of intermediate
    states: the intermediate index upper bound is max(n, m) + cutoff (inclusive).
    Larger `cutoff` values should improve accuracy but increase computational cost.
    This backend is intended for validation and exploration, not for final exact
    assembly.

    The expansion sums over intermediate oscillator states:
        Σ_{k=0}^{K} Σ_{l=0}^{K}
            ⟨n|D(β_left_op)|k⟩ ⟨k|G_{α,0}|l⟩ ⟨l|D(β_right_op)|m⟩
    where K = max(n, m) + cutoff (inclusive), and the displacement parameters
    are derived from the geometric factorization of the displaced Gaussian.

    Note: If `cutoff` is None, a default of 20 is used.
    """
    import math

    beta_left = math.sqrt(omega / 2.0) * center_left
    beta_right = math.sqrt(omega / 2.0) * center_right
    beta_gaussian = math.sqrt(omega / 2.0) * gaussian_center

    beta_left_op = beta_gaussian - beta_left
    beta_right_op = beta_right - beta_gaussian

    max_nm = max(n, m)
    if cutoff is None:
        cutoff = 20
    max_index = max_nm + cutoff

    total = 0.0
    for k in range(max_index + 1):
        left_factor = displacement_matrix_element(n, k, beta_left_op)
        if left_factor == 0.0:
            continue
        for l in range(max_index + 1):
            center_factor = origin_gaussian_matrix_element(k, l, omega, alpha)
            if center_factor == 0.0:
                continue
            right_factor = displacement_matrix_element(l, m, beta_right_op)
            total += left_factor * center_factor * right_factor

    return total


# Backward compatibility alias
# WARNING: This alias currently points to the truncated validation backend.
# It is not an exact displaced Gaussian implementation and may be replaced
# by an exact implementation in the future.
displaced_gaussian_matrix_element = displaced_gaussian_matrix_element_truncated


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
