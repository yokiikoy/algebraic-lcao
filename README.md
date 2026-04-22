# qmarg

`qmarg` is a small research prototype for comparing LCAO basis families on the
same one-dimensional two-center soft-Coulomb problem.

The current mainstream experiment is:

- reference: finite-difference solution of the 1D two-center soft-Coulomb Hamiltonian
- model 1: displaced harmonic-oscillator LCAO basis
- model 2: local monomial-Gaussian tower basis
- shared matrix assembly: real-space quadrature
- shared solver: symmetric orthogonalization for `H c = E S c`

Run the default convergence comparison:

```bash
python3 project/cli/run.py convergence
```

Run a single basis-size comparison:

```bash
python3 project/cli/run.py compare --basis-size 4
```

Check the first algebraic assembler against the existing real-space assembler:

```bash
python3 project/cli/run.py algebraic-check --basis-size 4 --omega 0.8
```

The planned Prolog integration is documented in
[`docs/prolog_plan.md`](docs/prolog_plan.md). Prolog is intended as a symbolic
formula generator and rule checker; Python remains responsible for numerical
evaluation, matrix assembly, eigensolvers, and convergence experiments.
