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


class PrologQueryError(RuntimeError):
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


_NONZERO_RE = re.compile(
    r"^nonzero\(target\((\d+)\),coeff\(source\((\d+)\),target\((\d+)\),denominator\((\d+)\)\)\)$"
)


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
    try:
        completed = subprocess.run(
            [swipl_executable(), "-q", "-s", str(prolog_file), "-g", goal],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PrologQueryError(
            "SWI-Prolog query failed: "
            f"goal={goal!r}, returncode={exc.returncode}, "
            f"stdout={exc.stdout!r}, stderr={exc.stderr!r}"
        ) from exc
    return parse_ladder_result(completed.stdout)


# ---------------------------------------------------------------------------
# Displacement finite-sum bridge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DisplacementTerm:
    p: int
    q: int
    ladder_coefficient: LadderCoefficient

    def evaluate_beta_factor(self, beta: float) -> float:
        """Evaluate beta^p / p! * (-beta)^q / q!."""
        return (
            (beta ** self.p / math.factorial(self.p))
            * ((-beta) ** self.q / math.factorial(self.q))
        )


@dataclass(frozen=True)
class DisplacementFiniteSum:
    terms: tuple[DisplacementTerm, ...]

    def evaluate(self, beta: float) -> float:
        """Evaluate <n | D(beta) | m> from the finite-sum representation."""
        prefactor = math.exp(-0.5 * beta ** 2)
        total = 0.0
        for term in self.terms:
            beta_factor = term.evaluate_beta_factor(beta)
            ladder_value = term.ladder_coefficient.evaluate()
            total += beta_factor * ladder_value
        return prefactor * total


_DISPLACEMENT_SUM_RE = re.compile(
    r"^displacement_sum\(prefactor\(exp_minus_half_beta_sq\),\[(.*)\]\)$"
)

_TERM_RE = re.compile(
    r"term\(p\((\d+)\),q\((\d+)\),ladder_coeff\(source\((\d+)\),target\((\d+)\),denominator\((\d+)\)\)\)"
)


def parse_displacement_finite_sum(text: str) -> DisplacementFiniteSum:
    stripped = text.strip()
    match = _DISPLACEMENT_SUM_RE.match(stripped)
    if match is None:
        raise ValueError(f"Unsupported Prolog displacement sum result: {text!r}")

    terms_text = match.group(1)
    if not terms_text:
        return DisplacementFiniteSum(terms=())

    terms = []
    for term_match in _TERM_RE.finditer(terms_text):
        p, q, source, target, denominator = (int(part) for part in term_match.groups())
        terms.append(
            DisplacementTerm(
                p=p,
                q=q,
                ladder_coefficient=LadderCoefficient(
                    source=source,
                    target=target,
                    denominator=denominator,
                ),
            )
        )

    return DisplacementFiniteSum(terms=tuple(terms))


def query_displacement_finite_sum(
    n: int,
    m: int,
    prolog_file: Path = DEFAULT_PROLOG_FILE,
) -> DisplacementFiniteSum:
    """Request the Prolog-generated finite-sum representation for <n | D(beta) | m>."""
    goal = f"emit_displacement_finite_sum({n},{m}),halt."
    try:
        completed = subprocess.run(
            [swipl_executable(), "-q", "-s", str(prolog_file), "-g", goal],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PrologQueryError(
            "SWI-Prolog query failed: "
            f"goal={goal!r}, returncode={exc.returncode}, "
            f"stdout={exc.stdout!r}, stderr={exc.stderr!r}"
        ) from exc
    return parse_displacement_finite_sum(completed.stdout)
