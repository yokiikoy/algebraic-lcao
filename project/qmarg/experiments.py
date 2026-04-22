from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qmarg.assembler import RealSpaceMatrixAssembler
from qmarg.domain import GridSpec, LcaoExperimentResult
from qmarg.optimization import (
    DisplacedHoBasisFactory,
    PolynomialGaussianBasisFactory,
    ScalarGridSearchOptimizer,
)
from qmarg.problems import TwoCenterSoftCoulombParams, TwoCenterSoftCoulombProblem
from qmarg.quadrature import GaussHermiteQuadrature
from qmarg.reference import FiniteDifferenceReferenceSolver
from qmarg.report import ComparisonSummary, ConvergenceRow
from qmarg.runner import LcaoRunner
from qmarg.solver import SymmetricOrthogonalizationSolver


@dataclass(frozen=True)
class ExperimentConfig:
    z: float = 1.0
    a: float = 1.5
    mu: float = 0.7
    grid_left: float = -12.0
    grid_right: float = 12.0
    grid_points: int = 1200
    quadrature_order: int = 200
    candidate_min: float = 0.2
    candidate_max: float = 2.0
    candidate_count: int = 31
    objective: str = "e1"


def default_problem(config: ExperimentConfig) -> TwoCenterSoftCoulombProblem:
    return TwoCenterSoftCoulombProblem(
        TwoCenterSoftCoulombParams(z=config.z, a=config.a, mu=config.mu)
    )


def default_runner(config: ExperimentConfig) -> LcaoRunner:
    return LcaoRunner(
        assembler=RealSpaceMatrixAssembler(
            quadrature=GaussHermiteQuadrature(order=config.quadrature_order)
        ),
        solver=SymmetricOrthogonalizationSolver(overlap_threshold=1e-10),
    )


def reference_eigenvalues(config: ExperimentConfig, num_states: int) -> np.ndarray:
    solver = FiniteDifferenceReferenceSolver(
        GridSpec(config.grid_left, config.grid_right, config.grid_points)
    )
    return solver.solve(default_problem(config), num_states)


def optimized_models(
    config: ExperimentConfig,
    basis_size: int,
    num_states: int,
) -> list[LcaoExperimentResult]:
    if basis_size % 2 != 0:
        raise ValueError("basis_size must be even because this is a two-center basis.")

    problem = default_problem(config)
    runner = default_runner(config)
    candidates = np.geomspace(
        config.candidate_min,
        config.candidate_max,
        config.candidate_count,
    )
    optimizer = ScalarGridSearchOptimizer(candidates=candidates, objective=config.objective)
    functions_per_center = basis_size // 2

    factories = [
        DisplacedHoBasisFactory(problem.params.a, functions_per_center),
        PolynomialGaussianBasisFactory(problem.params.a, functions_per_center),
    ]
    return [
        optimizer.optimize(factory, runner, problem, num_states)
        for factory in factories
    ]


def compare_basis_size(
    config: ExperimentConfig,
    basis_size: int,
    num_states: int = 2,
) -> ComparisonSummary:
    reference = reference_eigenvalues(config, num_states)
    return ComparisonSummary(
        reference_eigenvalues=reference,
        models=optimized_models(config, basis_size, num_states),
    )


def convergence_rows(
    config: ExperimentConfig,
    basis_sizes: list[int],
    num_states: int = 2,
) -> list[ConvergenceRow]:
    reference = reference_eigenvalues(config, num_states)
    rows: list[ConvergenceRow] = []
    for basis_size in basis_sizes:
        for model in optimized_models(config, basis_size, num_states):
            param_key = "omega" if model.model_name.startswith("displaced_ho") else "gamma"
            rows.append(
                ConvergenceRow(
                    basis_size=basis_size,
                    model_name=model.model_name,
                    opt_param=model.parameters[param_key],
                    condition_number=model.overlap_condition_number,
                    eigenvalues=model.eigenvalues,
                    reference_eigenvalues=reference,
                )
            )
    return rows
