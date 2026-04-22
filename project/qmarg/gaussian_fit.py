from __future__ import annotations

import numpy as np


def fit_soft_coulomb_gaussians(
    mu: float,
    count: int,
    rmax: float = 8.0,
    npts: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.linspace(0.0, rmax, npts)
    y = 1.0 / np.sqrt(r**2 + mu**2)
    exponents = np.geomspace(0.08, 4.0, count)

    design = np.exp(-np.outer(r**2, exponents))
    weights = 1.0 / (0.3 + r)
    weighted_design = design * weights[:, None]
    weighted_y = y * weights

    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    return exponents, coefficients


def gaussian_fit_errors(
    mu: float,
    exponents: np.ndarray,
    coefficients: np.ndarray,
    rmax: float = 6.0,
    npts: int = 500,
) -> tuple[float, float]:
    r = np.linspace(0.0, rmax, npts)
    exact = 1.0 / np.sqrt(r**2 + mu**2)
    fitted = np.zeros_like(r)
    for exponent, coefficient in zip(exponents, coefficients):
        fitted += coefficient * np.exp(-exponent * r**2)

    err = fitted - exact
    return float(np.sqrt(np.mean(err**2))), float(np.max(np.abs(err)))
