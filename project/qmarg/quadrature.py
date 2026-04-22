from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from numpy.polynomial.hermite import hermgauss


class RealLineQuadrature(Protocol):
    def nodes(self) -> np.ndarray:
        ...

    def weights_for_plain_integral(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class GaussHermiteQuadrature:
    order: int

    @cached_property
    def _nodes_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return hermgauss(self.order)

    def nodes(self) -> np.ndarray:
        nodes, _ = self._nodes_and_weights
        return nodes

    def weights_for_plain_integral(self) -> np.ndarray:
        nodes, weights = self._nodes_and_weights
        return weights * np.exp(nodes**2)
