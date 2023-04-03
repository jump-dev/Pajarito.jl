# Pajarito.jl

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
nondifferentiability. Pajarito may be more robust than Pavito on nonsmooth
problems.

## License

`Pajarito.jl` is licensed under the [MPL 2.0 license](https://github.com/jump-dev/Pajarito.jl/blob/master/LICENSE.md).

## Installation

Pajarito can be installed through the Julia package manager:
```julia
import Pkg
Pkg.add("Pajarito")
```

There are several convenient ways to model MICPs in Julia and access Pajarito.
[JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl) are algebraic modeling interfaces, while [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is a lower-level interface.

## MIP and continuous solvers

The algorithm implemented by Pajarito itself is relatively simple, and most of the hard work is performed by the MIP outer approximation (OA) solver and the continuous conic solver.
**The performance of Pajarito depends on these two types of solvers.**

The OA solver (typically a mixed-integer linear solver) is specified by the `oa_solver` option.
You must first load the Julia package that provides this solver, e.g. `using Gurobi`.
The continuous conic solver is specified by the `conic_solver` option.
See JuMP's [list of supported solvers](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers).

## Solver options

We list Pajarito's options below.

- `verbose::Bool` toggles printing
- `tol_feas::Float64` is the feasibility tolerance for conic constraints
- `tol_rel_gap::Float64` is the relative optimality gap tolerance
- `tol_abs_gap::Float64` is the absolute optimality gap tolerance
- `time_limit::Float64` sets the time limit (in seconds)
- `iteration_limit::Int` sets the iteration limit (for the iterative method)
- `use_iterative_method::Union{Nothing,Bool}` toggles the iterative algorithm; if `nothing' is specified, Pajarito defaults to the OA-solver-driven (single tree) algorithm if lazy callbacks are supported by the OA solver
- `use_extended_form::Bool` toggles the use of extended formulations (e.g. for the second-order cone)
- `solve_relaxation::Bool` toggles solution of the continuous conic relaxation
- `solve_subproblems::Bool` toggles solution of the continuous conic subproblems
- `use_init_fixed_oa::Bool` toggles initial fixed OA cuts
- `oa_solver::Union{Nothing,MOI.OptimizerWithAttributes}` is the OA solver
- `conic_solver::Union{Nothing,MOI.OptimizerWithAttributes}` is the conic solver

**Pajarito may require tuning of parameters to improve convergence.**
For example, it often helps to tighten the OA solver's integrality tolerance.
OA solver and conic solver options must be specified directly to those solvers.

Note: if `solve_subproblems` is true, Pajarito usually returns a solution constructed from one of the conic solver's feasible solutions; since the conic solver is not subject to the same feasibility tolerances as the OA solver, Pajarito's solution will not necessarily satisfy `tol_feas`.

## JuMP example

```julia
using JuMP, Pajarito, HiGHS, Hypatia

# setup solvers
oa_solver = optimizer_with_attributes(HiGHS.Optimizer,
    MOI.Silent() => true,
    "mip_feasibility_tolerance" => 1e-8,
    "mip_rel_gap" => 1e-6,
)
conic_solver = optimizer_with_attributes(Hypatia.Optimizer,
    MOI.Silent() => true,
)
opt = optimizer_with_attributes(Pajarito.Optimizer,
    "time_limit" => 60,
    "oa_solver" => oa_solver,
    "conic_solver" => conic_solver,
)

# setup model
model = Model(opt)
@variable(model, x, Int)
@variable(model, y)
@variable(model, z, Int)
@constraint(model, z <= 2.5)
@objective(model, Min, x + 2y)
@constraint(model, [z, x, y] in SecondOrderCone())

# solve
optimize!(model)
@show termination_status(model) # MOI.OPTIMAL
@show primal_status(model) # MOI.FEASIBLE_POINT
@show objective_value(model) # -1 - 2 * sqrt(3)
@show value(x) # -1
@show value(y) # -sqrt(3)
@show value(z) # 2
```

## Cone interface

Pajarito has a generic cone interface (see the [cones folder](src/Cones/)) that allows the user to add support for new convex cones.
To illustrate, in the experimental package [PajaritoExtras](https://github.com/chriscoey/PajaritoExtras.jl) we have extended Pajarito by adding support for several cones recognized by [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl) (a continuous conic solver with its own generic cone interface).
The examples folder of PajaritoExtras also contains many applied mixed-integer convex problems that are solved using Pajarito.

## Bug reports and support

Please report any issues via the Github [issue tracker](https://github.com/jump-dev/Pajarito.jl/issues).
All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc.
The [Optimization (Mathematical)](https://discourse.julialang.org/c/domain/opt) category on Discourse is appropriate for general discussion.

## References

If you find Pajarito useful in your work, we kindly request that you cite the following paper ([arXiv preprint](http://arxiv.org/abs/1808.05290)), which is recommended reading for advanced users:

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

Note this paper describes a legacy MathProgBase version of Pajarito, which is available on the [`mathprogbase` branch](https://github.com/jump-dev/Pajarito.jl/tree/mathprogbase) of this repository.
Starting with version 0.8.0, Pajarito only supports MathOptInterface.
