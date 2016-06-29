# Pajarito

Pajarito (**P**olyhedral **A**pproximation in **J**ulia: **A**utomatic **R**eformulations for **I**n**T**eger **O**ptimization) is a mixed-integer convex programming solver package written in [Julia](http://julialang.org/). Pajarito combines state-of-the-art convex and mixed-integer linear solvers by constructing sequential polyhedral approximations of the convex parts with a guaranteed finite-time convergence under minimal assumptions. Pajarito supports both **mixed-integer conic programming** and **smooth, derivative-based mixed-integer nonlinear programming**.

[![Build Status](https://travis-ci.org/mlubin/Pajarito.jl.svg?branch=master)](https://travis-ci.org/mlubin/Pajarito.jl) [![codecov.io](https://codecov.io/github/mlubin/Pajarito.jl/coverage.svg?branch=master)](https://codecov.io/github/mlubin/Pajarito.jl?branch=master)

## Installation

Pajarito can be installed through the Julia package manager:

```
julia> Pkg.add("Pajarito")
```

## Should I use Pajarito?

Pajarito is a solver for mixed-integer convex optimization problems. These are optimization problems where some variables are restricted to take discrete (e.g. binary) values, and the problem becomes convex when the discreteness constraints are relaxed. If your problem falls into this class, you may consider using Pajarito.

## Using Pajarito

The best way to access Pajarito is through [Convex.jl](https://github.com/JuliaOpt/Convex.jl), a disciplined convex programming (DCP) modeling language. Write a standard Convex.jl model with integer or binary variables and pass ``PajaritoSolver()`` (with some options; see below) to Convex.jl's ``solve!`` method. In the context of DCP, we call Pajarito a mixed-integer DCP (MIDCP) solver, the first published MIDCP solver in existence to our knowledge. When used through Convex.jl, Pajarito acts as a mixed-integer conic solver using the algorithm described in our paper cited below.

Pajarito is also accessible through [JuMP](https://github.com/JuliaOpt/JuMP.jl) as a mixed-integer convex nonlinear solver by setting the ``solver=PajaritoSolver()`` option in JuMP's ``Model()`` constructor. When used in this way, Pajarito is analogous to [Bonmin](https://projects.coin-or.org/Bonmin) and will perform similarly, with the primary advantage of being able to easily swap-in various mixed-integer and convex subproblem solvers which Bonmin does not support. Note that Pajarito does not verify convexity of derivative-based input and may give incorrect answers to nonconvex problems.

**We recommend Convex.jl over JuMP as input to Pajarito; a problem expressed in in Convex.jl form may solve faster than an identical problem expressed using JuMP.** This is because Convex.jl automatically transforms problems into mixed-integer conic form while JuMP provides problems to Pajarito in the more traditional derivative-based form. In our paper cited below, we argue that mixed-integer conic form is a superior representation. Nevertheless, this question remains an active area of research and we encourage users to experiment with multiple formulations to see which works best. [Hijazi et al.](http://www.optimization-online.org/DB_FILE/2011/06/3050.pdf) suggest manual reformulation techniques which achieve many of the algorithmic benefits of conic form.

Pajarito may be accessed from outside Julia by using the experimental [cmpb](https://github.com/mlubin/cmpb) interface which provides a C API to the low-level conic input format.

## Subproblem solvers

The algorithm implemented by Pajarito itself is relatively simple, and most of the hard work is performed by subproblem solvers. Pajarito requires two different subproblem solvers, one for mixed-integer linear problems and one for convex subproblems. **The performance of Pajarito depends on the subproblem solvers.** For best performance, use commercial solvers.

The mixed-integer linear solver is specified by using the `mip_solver` option to `PajaritoSolver`, e.g., `PajaritoSolver(mip_solver=CplexSolver())`. You must first load the Julia package which provides the mixed-integer linear solver, e.g., with `using CPLEX`.

The convex subproblem solver is specified by using the `cont_solver` option, e.g., `PajaritoSolver(cont_solver=IpoptSolver())`. When given input in derivative-based nonlinear form, Pajarito requires a derivative-based nonlinear solver, e.g., [Ipopt](https://projects.coin-or.org/Ipopt) or [KNITRO](http://www.ziena.com/knitro.htm). When given input in conic form, the convex subproblem solver can be *either* a conic solver like [ECOS](https://github.com/JuliaOpt/ECOS.jl) *or* a derivative-based solver like Ipopt. If a derivative-based solver is provided in this case, then Pajarito will go ahead and automatically use it to solve the conic subproblems by using the [ConicNonlinearBridge](https://github.com/mlubin/ConicNonlinearBridge.jl) package. Note that using derivative-based solvers for conic problems can cause numerical instability because conic problems are not always smooth.

All solvers can have their parameters specified through their corresponding Julia interfaces. For example, you probably should turn off the output of the subproblem solvers, e.g., by using `IpoptSolver(print_level=0)`.

## Pajarito solver options

The following options can be passed to `PajaritoSolver()` to modify its behavior:

  * `log_level::Int`              # Verbosity flag: 1 for minimal OA iteration and solve statistics, 2 for including cone summary information, 3 for running commentary
  * `mip_solver_drives::Bool`     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
  * `soc_in_mip::Bool`            # Use SOC/SOCRotated cones in the MIP outer approximation model (if MIP solver supports MISOCP)
  * `disagg_soc::Bool`            # Disaggregate SOC/SOCRotated cones in the MIP only (if solver is conic)
  * `drop_dual_infeas::Bool`      # Do not add cuts from dual cone infeasible dual vectors
  * `proj_dual_infeas::Bool`      # Project dual cone infeasible dual vectors onto dual cone boundaries
  * `proj_dual_feas::Bool`        # Project dual cone strictly feasible dual vectors onto dual cone boundaries
  * `mip_solver::MathProgBase.AbstractMathProgSolver` # MIP solver
  * `cont_solver::MathProgBase.AbstractMathProgSolver` # Continuous solver
  * `timeout::Float64`            # Time limit for outer approximation algorithm not including initial load (in seconds)
  * `rel_gap::Float64`            # Relative optimality gap termination condition
  * `zero_tol::Float64`           # Tolerance for setting small absolute values to zeros

**Pajarito is not yet numerically robust and may require tuning of parameters to improve convergence.** If the default parameters don't work for you, please let us know.

## Bug reports and support

Please report any issues via the Github **[issue tracker]**. All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc. The **[julia-opt]** mailing list is appropriate for general discussion.

[issue tracker]: https://github.com/mlubin/Pajarito.jl/issues
[julia-opt]: https://groups.google.com/forum/#!forum/julia-opt

## We need your challenging MICP problems

Mixed-integer convex programming is an active area of research, and we are seeking out hard benchmark instances. Please get in touch either by opening an issue or privately if you would like to share any hard instances to be used as benchmarks in future work. Challenging problems will help us determine how to improve Pajarito.

## References

If you find Pajarito useful in your work, we kindly request that you cite the following [paper](http://arxiv.org/abs/1511.06710):

    @article{LubinYamangilBentVielma2016,
    title = {Extended Formulations in Mixed-integer Convex Programming},
    author = {Miles Lubin, Emre Yamangil, Russell Bent, Juan Pablo Vielma},
    journal = {arXiv:1511.06710 [math.OC]},
    year = {2016},
    url = {http://arxiv.org/abs/1511.06710}
    }

The paper describes the motivation of Pajarito and is recommended reading for advanced users.
