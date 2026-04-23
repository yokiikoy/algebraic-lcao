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


def displaced_gaussian_matrix_element(
    n: int,
    center_left: float,
    m: int,
    center_right: float,
    omega: float,
    alpha: float,
    gaussian_center: float,
    cutoff: int | None = None,
) -> float:
    """Return ⟨n,A|exp(-α (x-C)²)|m,B⟩ using decomposition formula.
    
    Decomposition formula from docs/su11_gaussian_operator.md:
    ⟨n,A|G_{α,C}|m,B⟩ = ⟨n| D(β_C - β_A) G_{α,0} D(β_B - β_C) |m⟩
    where β_X = sqrt(ω/2) * X and G_{α,0} = exp(-α x²).
    
    Important: This is a *truncated* double-sum realization of the decomposition.
    It is **not** an exact closed-form displaced backend. The intermediate states
    are summed only up to a finite cutoff, so the result is approximate and
    intended mainly for validation and exploratory use.
    
    Implementation uses double sum over intermediate states k, l:
    ∑_k ∑_l ⟨n| D(β_C - β_A) |k⟩ ⟨k| G_{α,0} |l⟩ ⟨l| D(β_B - β_C) |m⟩
    truncated to k, l < max(n, m) + cutoff.
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
    for k in range(max_index):
        left_factor = displacement_matrix_element(n, k, beta_left_op)
        if left_factor == 0.0:
            continue
        for l in range(max_index):
            center_factor = origin_gaussian_matrix_element(k, l, omega, alpha)
            if center_factor == 0.0:
                continue
            right_factor = displacement_matrix_element(l, m, beta_right_op)
            total += left_factor * center_factor * right_factor
    
    return total


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
