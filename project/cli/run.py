from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmarg.algebraic_assembler import AlgebraicGaussianExpansionAssembler
from qmarg.assembler import RealSpaceMatrixAssembler
from qmarg.basis import DisplacedHoBasis
from qmarg.experiments import ExperimentConfig, compare_basis_size, convergence_rows
from qmarg.gaussian_fit import fit_soft_coulomb_gaussians, gaussian_fit_errors
from qmarg.problems import GaussianExpansionProblem, two_center_gaussian_expansion_terms
from qmarg.quadrature import GaussHermiteQuadrature
from qmarg.report import convergence_table
from qmarg.solver import SymmetricOrthogonalizationSolver


def parse_basis_sizes(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 1D two-center soft-Coulomb LCAO comparison experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--z", type=float, default=1.0)
        command.add_argument("--a", type=float, default=1.5)
        command.add_argument("--mu", type=float, default=0.7)
        command.add_argument("--grid-left", type=float, default=-12.0)
        command.add_argument("--grid-right", type=float, default=12.0)
        command.add_argument("--grid-points", type=int, default=1200)
        command.add_argument("--quadrature-order", type=int, default=200)
        command.add_argument("--candidate-min", type=float, default=0.2)
        command.add_argument("--candidate-max", type=float, default=2.0)
        command.add_argument("--candidate-count", type=int, default=31)
        command.add_argument("--objective", choices=["e1", "e2", "sum12"], default="e1")

    compare = subparsers.add_parser("compare", help="Compare both basis families at one size.")
    add_common_options(compare)
    compare.add_argument("--basis-size", type=int, default=4)

    convergence = subparsers.add_parser(
        "convergence",
        help="Compare both basis families over a basis-size sequence.",
    )
    add_common_options(convergence)
    convergence.add_argument("--basis-sizes", default="2,4,6,8")

    algebraic = subparsers.add_parser(
        "algebraic-check",
        help="Compare real-space and algebraic assemblers for a Gaussian-expanded potential.",
    )
    add_common_options(algebraic)
    algebraic.add_argument("--basis-size", type=int, default=4)
    algebraic.add_argument("--omega", type=float, default=0.8)
    algebraic.add_argument("--gaussians", type=int, default=4)

    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        z=args.z,
        a=args.a,
        mu=args.mu,
        grid_left=args.grid_left,
        grid_right=args.grid_right,
        grid_points=args.grid_points,
        quadrature_order=args.quadrature_order,
        candidate_min=args.candidate_min,
        candidate_max=args.candidate_max,
        candidate_count=args.candidate_count,
        objective=args.objective,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    if args.command == "compare":
        print(compare_basis_size(config, args.basis_size).to_text())
        return

    if args.command == "algebraic-check":
        exponents, weights = fit_soft_coulomb_gaussians(config.mu, args.gaussians)
        fit_rmse, fit_max = gaussian_fit_errors(config.mu, exponents, weights)
        terms = two_center_gaussian_expansion_terms(
            center_distance=config.a,
            z=config.z,
            exponents=exponents,
            weights=weights,
        )
        problem = GaussianExpansionProblem(terms)
        basis = DisplacedHoBasis(
            center_distance=config.a,
            functions_per_center=args.basis_size // 2,
            omega=args.omega,
        )

        real_h, real_s = RealSpaceMatrixAssembler(
            GaussHermiteQuadrature(config.quadrature_order)
        ).assemble(problem, basis)
        alg_h, alg_s = AlgebraicGaussianExpansionAssembler().assemble(problem, basis)

        solver = SymmetricOrthogonalizationSolver()
        real_e = solver.solve(real_h, real_s, 2).eigenvalues
        alg_e = solver.solve(alg_h, alg_s, 2).eigenvalues

        print("Gaussian-expanded soft-Coulomb fit:")
        print(f"  gaussian_count = {args.gaussians}")
        print(f"  fit_rmse = {fit_rmse:.6e}")
        print(f"  fit_max_abs = {fit_max:.6e}")
        print("")
        print("Assembler consistency:")
        print(f"  max_abs_delta_H = {np.max(np.abs(real_h - alg_h)):.6e}")
        print(f"  max_abs_delta_S = {np.max(np.abs(real_s - alg_s)):.6e}")
        print("")
        print("Eigenvalues for Gaussian-expanded Hamiltonian:")
        for i, (er, ea) in enumerate(zip(real_e, alg_e), start=1):
            print(f"  E[{i}] real={er:.10f} algebraic={ea:.10f} delta={ea - er:+.6e}")
        return

    rows = convergence_rows(config, parse_basis_sizes(args.basis_sizes))
    print("Reference problem: 1D two-center soft Coulomb")
    print(f"Optimization objective: {config.objective}")
    print("")
    print(convergence_table(rows))


if __name__ == "__main__":
    main()
