# Pajarito.jl

[![Build Status](https://github.com/jump-dev/Pajarito.jl/workflows/CI/badge.svg?branch=main)](https://github.com/jump-dev/Pajarito.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/Pajarito.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jump-dev/Pajarito.jl)


Pajarito is a **mixed-integer convex programming** (MICP) solver package written
in [Julia](http://julialang.org/).

MICP problems are convex except for restrictions that some variables take binary
or integer values.

Pajarito solves MICP problems in conic form, by constructing sequential
polyhedral outer approximations of the conic feasible set.

The underlying algorithm has theoretical finite-time convergence under
reasonable assumptions.

Pajarito accesses state-of-the-art MILP solvers and continuous conic solvers
through [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl).

## Pavito

For algorithms that use a derivative-based nonlinear programming (NLP) solver
(for example, Ipopt) instead of a conic solver, use [Pavito](https://github.com/jump-dev/Pavito.jl).

Pavito is a convex mixed-integer nonlinear programming (convex MINLP) solver.
Because Pavito relies on gradient cuts, it can fail near points of
non-differentiability. Pajarito may be more robust than Pavito on non-smooth
problems.

## License

`Pajarito.jl` is licensed under the [MPL 2.0 license](https://github.com/jump-dev/Pajarito.jl/blob/master/LICENSE.md).

## Installation

Install Pajarito using `Pkg.add`:
```julia
import Pkg
Pkg.add("Pajarito")
```

## MIP and continuous solvers

The algorithm implemented by Pajarito itself is relatively simple, and most of
the hard work is performed by the MIP outer approximation (OA) solver and the
continuous conic solver.

Therefore, in addition to installing Pajarito, you must also install a
mixed-integer linear programming solver and a continuous conic solver.

**The performance of Pajarito depends on these two types of solvers.**

The OA solver (typically a mixed-integer linear solver) is specified by the
`oa_solver` option. You must first load the Julia package that provides this
solver, for example, `using Gurobi`. The continuous conic solver is specified by
the `conic_solver` option.

See JuMP's [list of supported solvers](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers).

## Use with JuMP

To use Pajarito with JuMP, use:
```julia
using JuMP, Pajarito, HiGHS, Hypatia
model = Model(
    optimizer_with_attributes(
        Pajarito.Optimizer,
        "oa_solver" => optimizer_with_attributes(
            HiGHS.Optimizer,
            MOI.Silent() => true,
            "mip_feasibility_tolerance" => 1e-8,
            "mip_rel_gap" => 1e-6,
        ),
        "conic_solver" =>
            optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => true),
    )
)
set_attribute(model, "time_limit", 60)
```

## Options

We list Pajarito's options below.

- `verbose::Bool` toggles printing
- `tol_feas::Float64` is the feasibility tolerance for conic constraints
- `tol_rel_gap::Float64` is the relative optimality gap tolerance
- `tol_abs_gap::Float64` is the absolute optimality gap tolerance
- `time_limit::Float64` sets the time limit (in seconds)
- `iteration_limit::Int` sets the iteration limit (for the iterative method)
- `use_iterative_method::Union{Nothing,Bool}` toggles the iterative algorithm;
  if `nothing` is specified, Pajarito defaults to the OA-solver-driven (single
  tree) algorithm if lazy callbacks are supported by the OA solver
- `use_extended_form::Bool` toggles the use of extended formulations for the
  second-order cone
- `solve_relaxation::Bool` toggles solution of the continuous conic relaxation
- `solve_subproblems::Bool` toggles solution of the continuous conic subproblems
- `use_init_fixed_oa::Bool` toggles initial fixed OA cuts
- `oa_solver::Union{Nothing,MOI.OptimizerWithAttributes}` is the OA solver
- `conic_solver::Union{Nothing,MOI.OptimizerWithAttributes}` is the conic solver

**Pajarito may require tuning of parameters to improve convergence.**

For example, it often helps to tighten the OA solver's integrality tolerance.
OA solver and conic solver options must be specified directly to those solvers.

Note: if `solve_subproblems` is true, Pajarito usually returns a solution
constructed from one of the conic solver's feasible solutions; since the conic
solver is not subject to the same feasibility tolerances as the OA solver,
Pajarito's solution will not necessarily satisfy `tol_feas`.

## Cone interface

Pajarito has a generic cone interface (see the [cones folder](https://github.com/jump-dev/Pajarito.jl/tree/main/src/Cones)
that allows the user to add support for new convex cones.

To illustrate, in the experimental package [PajaritoExtras](https://github.com/chriscoey/PajaritoExtras.jl)
we have extended Pajarito by adding support for several cones recognized by
[Hypatia.jl](https://github.com/chriscoey/Hypatia.jl) (a continuous conic solver
with its own generic cone interface).

The examples folder of PajaritoExtras also contains many applied mixed-integer
convex problems that are solved using Pajarito.

## Bug reports and support

Please report any issues via the GitHub
[issue tracker](https://github.com/jump-dev/Pajarito.jl/issues).

All types of issues are welcome and encouraged; this includes bug reports,
documentation typos, and feature requests. The [Optimization (Mathematical)](https://discourse.julialang.org/c/domain/opt)
category on Discourse is appropriate for general discussion.

## References

If you find Pajarito useful in your work, we kindly request that you cite the
following paper ([arXiv preprint](http://arxiv.org/abs/1808.05290)), which is
recommended reading for advanced users:

```
@article{CoeyLubinVielma2020,
    title={Outer approximation with conic certificates for mixed-integer convex problems},
    author={Coey, Chris and Lubin, Miles and Vielma, Juan Pablo},
    journal={Mathematical Programming Computation},
    volume={12},
    number={2},
    pages={249--293},
    year={2020},
    publisher={Springer}
}
```

Note this paper describes a legacy MathProgBase version of Pajarito, which is
available on the [`mathprogbase` branch](https://github.com/jump-dev/Pajarito.jl/tree/mathprogbase)
of this repository. Starting with version v0.8.0, Pajarito supports
MathOptInterface instead of MathProgBase.
