# Pajarito

Pajarito is a **mixed-integer convex programming** (MICP) solver package written in [Julia](http://julialang.org/). MICP problems are convex except for restrictions that some variables take binary or integer values.

Pajarito solves MICP problems in conic form, by constructing sequential polyhedral outer-approximations of the conic feasible set. The underlying algorithm has theoretical finite-time convergence under reasonable assumptions. 
Pajarito accesses state-of-the-art MILP solvers and continuous conic solvers through MathOptInterface. 

For algorithms that use a derivative-based nonlinear programming (NLP) solver (e.g. Ipopt) instead of a conic solver, use [Pavito](https://github.com/jump-dev/Pavito.jl). Pavito is a convex mixed-integer nonlinear programming (convex MINLP) solver. As Pavito relies on gradient cuts, it can fail near points of nondifferentiability. Pajarito may be more robust than Pavito on nonsmooth problems.

## Installation

Pajarito can be installed through the Julia package manager:
```
julia> Pkg.add("Pajarito")
```

## Usage

There are several convenient ways to model MICPs in Julia and access Pajarito.
[JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl) are algebraic modeling interfaces, while [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) is a lower-level interface.

## MIP and continuous solvers

The algorithm implemented by Pajarito itself is relatively simple, and most of the hard work is performed by the MIP solver and the continuous conic solver. **The performance of Pajarito depends on these two types of solvers.** 

The mixed-integer (outer approximation) solver is specified by using the `oa_solver` option. You must first load the Julia package which provides the mixed-integer solver, e.g. `using Gurobi`. 
The continuous conic solver is specified by using the `conic_solver` option. 
See JuMP's [list of supported solvers](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers).
MIP and continuous solver options must be specified through their corresponding Julia interfaces.

## Bug reports and support

Please report any issues via the Github **[issue tracker]**. All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc. The **[Optimization (Mathematical)]** category on Discourse is appropriate for general discussion.

[issue tracker]: https://github.com/mlubin/Pajarito.jl/issues
[Optimization (Mathematical)]: https://discourse.julialang.org/c/domain/opt

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