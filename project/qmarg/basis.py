from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math

import numpy as np


class BasisFunctionSet(Protocol):
    def size(self) -> int:
        ...

    def values(self, x: np.ndarray) -> np.ndarray:
        ...

    def second_derivatives(self, x: np.ndarray) -> np.ndarray:
        ...

    def parameter_dict(self) -> dict[str, float]:
        ...

    def model_name(self) -> str:
        ...


def hermite_phys(n: int, x: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x

    hm2 = np.ones_like(x)
    hm1 = 2.0 * x
    for k in range(1, n):
        h = 2.0 * x * hm1 - 2.0 * k * hm2
        hm2, hm1 = hm1, h
    return hm1


def ho_psi(n: int, x: np.ndarray, omega: float) -> np.ndarray:
    xi = np.sqrt(omega) * x
    norm = (omega / math.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))
    return norm * hermite_phys(n, xi) * np.exp(-0.5 * xi**2)


@dataclass(frozen=True)
class DisplacedHoBasis:
    center_distance: float
    functions_per_center: int
    omega: float

    def size(self) -> int:
        return 2 * self.functions_per_center

    def states(self) -> list[tuple[int, float]]:
        states = []
        for center in (-self.center_distance, self.center_distance):
            for n in range(self.functions_per_center):
                states.append((n, center))
        return states

    def model_name(self) -> str:
        return f"displaced_ho_{self.size()}"

    def parameter_dict(self) -> dict[str, float]:
        return {
            "a": self.center_distance,
            "omega": float(self.omega),
            "functions_per_center": float(self.functions_per_center),
        }

    def values(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for center in (-self.center_distance, self.center_distance):
            y = x - center
            for n in range(self.functions_per_center):
                rows.append(ho_psi(n, y, self.omega))
        return np.vstack(rows)

    def second_derivatives(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for center in (-self.center_distance, self.center_distance):
            y = x - center
            for n in range(self.functions_per_center):
                psi = ho_psi(n, y, self.omega)
                d2 = (self.omega**2 * y**2 - 2.0 * self.omega * (n + 0.5)) * psi
                rows.append(d2)
        return np.vstack(rows)


@dataclass(frozen=True)
class MonomialGaussianTowerBasis:
    """Local tower y^n exp(-gamma y^2) at each center.

    This is intentionally not a general contracted Gaussian LCAO basis. For a
    fixed `gamma`, it spans the same local polynomial-Gaussian tower as a
    harmonic-oscillator basis with the matching width.
    """

    center_distance: float
    functions_per_center: int
    gamma: float

    def size(self) -> int:
        return 2 * self.functions_per_center

    def model_name(self) -> str:
        return f"monomial_gaussian_tower_{self.size()}"

    def parameter_dict(self) -> dict[str, float]:
        return {
            "a": self.center_distance,
            "gamma": float(self.gamma),
            "functions_per_center": float(self.functions_per_center),
        }

    def values(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for center in (-self.center_distance, self.center_distance):
            y = x - center
            base = np.exp(-self.gamma * y**2)
            for power in range(self.functions_per_center):
                rows.append((y**power) * base)
        return np.vstack(rows)

    def second_derivatives(self, x: np.ndarray) -> np.ndarray:
        rows = []
        g = self.gamma
        for center in (-self.center_distance, self.center_distance):
            y = x - center
            base = np.exp(-g * y**2)
            for n in range(self.functions_per_center):
                term = -2.0 * g * (2 * n + 1) * y**n + 4.0 * g**2 * y ** (n + 2)
                if n >= 2:
                    term = term + n * (n - 1) * y ** (n - 2)
                rows.append(term * base)
        return np.vstack(rows)
