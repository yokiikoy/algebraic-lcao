from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROLOG_FILE = REPO_ROOT / "priv" / "prolog" / "main.pl"


class PrologUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LadderCoefficient:
    source: int
    target: int
    denominator: int

    def evaluate(self) -> float:
        return (
            math.sqrt(math.factorial(self.source) * math.factorial(self.target))
            / math.factorial(self.denominator)
        )


@dataclass(frozen=True)
class LadderMatrixElement:
    target: int | None
    coefficient: LadderCoefficient | None

    def is_zero(self) -> bool:
        return self.coefficient is None

    def evaluate(self) -> float:
        if self.coefficient is None:
            return 0.0
        return self.coefficient.evaluate()


_NONZERO_RE = re.compile(r"^nonzero\((\d+),coeff\((\d+),(\d+),(\d+)\)\)$")


def swipl_executable() -> str:
    executable = shutil.which("swipl")
    if executable is None:
        raise PrologUnavailable("SWI-Prolog executable `swipl` was not found.")
    return executable


def parse_ladder_result(text: str) -> LadderMatrixElement:
    stripped = text.strip()
    if stripped == "zero":
        return LadderMatrixElement(target=None, coefficient=None)

    match = _NONZERO_RE.match(stripped)
    if match is None:
        raise ValueError(f"Unsupported Prolog ladder result: {text!r}")

    target, source, coeff_target, denominator = (int(part) for part in match.groups())
    if target != coeff_target:
        raise ValueError(f"Inconsistent target in Prolog ladder result: {text!r}")

    return LadderMatrixElement(
        target=target,
        coefficient=LadderCoefficient(
            source=source,
            target=coeff_target,
            denominator=denominator,
        ),
    )


def query_ladder_matrix_element(
    n: int,
    p: int,
    q: int,
    m: int,
    prolog_file: Path = DEFAULT_PROLOG_FILE,
) -> LadderMatrixElement:
    goal = f"emit_me_ladder({n},{p},{q},{m}),halt."
    completed = subprocess.run(
        [swipl_executable(), "-q", "-s", str(prolog_file), "-g", goal],
        check=True,
        text=True,
        capture_output=True,
    )
    return parse_ladder_result(completed.stdout)
