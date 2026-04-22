from __future__ import annotations

import math


def origin_gaussian_matrix_element_su11(
    n: int,
    m: int,
    omega: float,
    alpha: float,
) -> float:
    """Return <n|exp(-alpha x^2)|m> for the centered oscillator basis.

    This is an independent centered backend for the non-unitary Gaussian
    multiplication operator. It uses the equivalent SU(1,1)/Gaussian generating
    function finite sum, not the shifted-moment backend in `fock.py`.

    With y = sqrt(omega) x and g = alpha / omega:

        exp(-alpha x^2) = exp(-g y^2)

    The Hermite generating integral reduces to coefficients of

        exp(A s^2 + A t^2 + B s t)

    where A = -g/(1+g) and B = 2/(1+g). This gives a finite parity-preserving
    sum for each pair (n,m).
    """
    if n < 0 or m < 0:
        return 0.0
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if (n + m) % 2 == 1:
        return 0.0

    g = alpha / omega
    scale = 1.0 + g
    a_coeff = -g / scale
    b_coeff = 2.0 / scale

    coeff_sum = 0.0
    for k in range(min(n, m) + 1):
        if (n - k) % 2 != 0 or (m - k) % 2 != 0:
            continue

        i = (n - k) // 2
        j = (m - k) // 2
        coeff_sum += (
            a_coeff**i
            / math.factorial(i)
            * a_coeff**j
            / math.factorial(j)
            * b_coeff**k
            / math.factorial(k)
        )

    normalization = (
        math.sqrt(math.factorial(n) * math.factorial(m))
        / math.sqrt(2.0 ** (n + m))
        / math.sqrt(scale)
    )
    return normalization * coeff_sum
