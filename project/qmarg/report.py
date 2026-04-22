from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmarg.domain import LcaoExperimentResult


@dataclass(frozen=True)
class ComparisonSummary:
    reference_eigenvalues: np.ndarray
    models: list[LcaoExperimentResult]

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append("Reference eigenvalues:")
        for i, e in enumerate(self.reference_eigenvalues, start=1):
            lines.append(f"  E_ref[{i}] = {e:.10f}")
        lines.append("")

        for model in self.models:
            lines.append(f"Model: {model.model_name}")
            lines.append(f"  basis_size = {model.basis_size}")
            lines.append(f"  params = {model.parameters}")
            lines.append(f"  overlap_condition_number = {model.overlap_condition_number:.6e}")
            for i, e in enumerate(model.eigenvalues, start=1):
                err = e - self.reference_eigenvalues[i - 1]
                lines.append(f"  E[{i}] = {e:.10f}    err = {err:+.6e}")
            lines.append("")

        return "\n".join(lines)


@dataclass(frozen=True)
class ConvergenceRow:
    basis_size: int
    model_name: str
    opt_param: float
    condition_number: float
    eigenvalues: np.ndarray
    reference_eigenvalues: np.ndarray


def convergence_table(rows: list[ConvergenceRow]) -> str:
    lines = [
        "basis model                    opt_param       cond(S)       "
        "E1             err1          E2             err2"
    ]
    for row in rows:
        e1, e2 = row.eigenvalues[:2]
        r1, r2 = row.reference_eigenvalues[:2]
        lines.append(
            f"{row.basis_size:>5d} {row.model_name:<22s} "
            f"{row.opt_param:>10.6f}  {row.condition_number:>11.4e}  "
            f"{e1:>13.10f} {e1 - r1:>+12.6e}  "
            f"{e2:>13.10f} {e2 - r2:>+12.6e}"
        )
    return "\n".join(lines)
