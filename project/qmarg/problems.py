from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TwoCenterSoftCoulombParams:
    z: float
    a: float
    mu: float


class Potential(Protocol):
    def value(self, x: np.ndarray) -> np.ndarray:
        ...


class HamiltonianProblem(Protocol):
    def potential(self) -> Potential:
        ...


@dataclass(frozen=True)
class TwoCenterSoftCoulombPotential:
    params: TwoCenterSoftCoulombParams

    def value(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        return (
            -p.z / np.sqrt((x - p.a) ** 2 + p.mu**2)
            - p.z / np.sqrt((x + p.a) ** 2 + p.mu**2)
        )


@dataclass(frozen=True)
class TwoCenterSoftCoulombProblem:
    params: TwoCenterSoftCoulombParams

    def potential(self) -> Potential:
        return TwoCenterSoftCoulombPotential(self.params)


@dataclass(frozen=True)
class GaussianPotentialTerm:
    coefficient: float
    exponent: float
    center: float


@dataclass(frozen=True)
class GaussianExpansionPotential:
    terms: tuple[GaussianPotentialTerm, ...]

    def value(self, x: np.ndarray) -> np.ndarray:
        total = np.zeros_like(x, dtype=float)
        for term in self.terms:
            total += term.coefficient * np.exp(-term.exponent * (x - term.center) ** 2)
        return total


@dataclass(frozen=True)
class GaussianExpansionProblem:
    terms: tuple[GaussianPotentialTerm, ...]

    def potential(self) -> Potential:
        return GaussianExpansionPotential(self.terms)


def two_center_gaussian_expansion_terms(
    center_distance: float,
    z: float,
    exponents: np.ndarray,
    weights: np.ndarray,
) -> tuple[GaussianPotentialTerm, ...]:
    terms: list[GaussianPotentialTerm] = []
    for center in (-center_distance, center_distance):
        for exponent, weight in zip(exponents, weights):
            terms.append(
                GaussianPotentialTerm(
                    coefficient=-float(z) * float(weight),
                    exponent=float(exponent),
                    center=float(center),
                )
            )
    return tuple(terms)
