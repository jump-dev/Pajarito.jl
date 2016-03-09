# Pajarito

Pajarito (Polyhedral Approximation in Julia: Automatic Reformulations for InTeger Optimization) is a mixed-integer convex programming solver package in [Julia](http://julialang.org/) that currently implements an out-of-the-box polyhedral outer approximation algorithm under src/nonlinear.jl and a polyhedral outer approximation and branch and cut for mixed-integer conic programming algorithm under src/conic.jl.

[![Build Status](https://travis-ci.org/mlubin/Pajarito.jl.svg?branch=master)](https://travis-ci.org/mlubin/Pajarito.jl) [![codecov.io](https://codecov.io/github/mlubin/Pajarito.jl/coverage.svg?branch=master)](https://codecov.io/github/mlubin/Pajarito.jl?branch=master)

## Installation

Pajarito can be installed through Julia:

```
julia> Pkg.clone("https://github.com/mlubin/Pajarito.jl.git")
```

It is recommended to install [ConicNonlinearBridge.jl](https://github.com/mlubin/ConicNonlinearBridge.jl) for the conic programming subproblems in case of numerical instability. See test/ folder for examples.

## Supported solvers

Pajarito requires a mixed-integer linear programming solver and a nonlinear programming solver for src/nonlinear.jl or a conic solver for src/conic.jl. 

For MILP solver, it currently supports only [CPLEX](http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/) and [GuRoBi](http://www.gurobi.com/) for the branch and cut algorithm for mixed-integer conic programming problems. Pajarito further supports [GLPK](http://www.gnu.org/software/glpk/) for polyhedral outer approximation under both src/nonlinear.jl and src/conic.jl. MILP solver can be specified through the `mip_solver` flag, i.e. `PajaritoSolver(mip_solver=CplexSolver())`.

For NLP solver, Pajarito has been tested with [Ipopt](https://projects.coin-or.org/Ipopt), [KNITRO](http://www.ziena.com/knitro.htm). NLP solver can be specified through `cont_solver` flag, i.e. `PajaritoSolver(cont_solver=IpoptSolver())`.

For Conic solver, Pajarito has been tested with [SCS](https://github.com/cvxgrp/scs), [ECOS](https://github.com/ifa-ethz/ecos), and furthermore ConicNonlinearBridge for flexible choice of a supported NLP solver (Ipopt and KNITRO). Conic solver can be specified through `cont_solver` flag, i.e. `PajaritoSolver(cont_solver=KnitroSolver())`.

All solvers can have their parameters specified through their corresponding Julia interfaces.

## Supported problem classes

Pajarito supports mixed-integer smooth convex programming problems through [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) (this functionality is similar to that of [Bonmin](https://projects.coin-or.org/Bonmin)) and mixed-integer disciplined convex programming problems through [Convex.jl](https://github.com/JuliaOpt/Convex.jl).

## Supported algorithms

Pajarito supports two algorithms for the mixed-integer conic programming problems. BC for a branch and cut algorithm and OA for polyhedral outer approximation algorithm. They can be specified as strings through the `algorithm` flag, i.e. `PajaritoSolver(algorithm="OA")`.

## Solver options

  * `verbose::Int`                : Verbosity level flag
  * `algorithm::String`           : Choice of algorithm: "OA" or "BC"
  * `mip_solver`                  : Choice of MILP solver
  * `cont_solver`                 : Choice of Conic solver
  * `opt_tolerance`               : Relatice optimality tolerance
  * `acceptable_opt_tolerance`    : Acceptable optimality tolerance if separation fails
  * `time_limit`                  : Time limit
  * `cut_switch`                  : Cut level for OA
  * `socp_disaggregator::Bool`    : SOCP disaggregator for SOC constraints
  * `instance::AbstractString`    : Path to instance

## Citing

If you find Pajarito useful in your work, we kindly request that you cite the following paper:

    @article{LubinYamangilBentVielma2016,
    title = {Extended Formulations in Mixed-integer Convex Programming},
    author = {Miles Lubin, Emre Yamangil, Russell Bent, Juan Pablo Vielma},
    journal = {arXiv:1511.06710 [math.OC]},
    year = {2016},
    url = {http://arxiv.org/abs/1511.06710}
    }

A preprint of this paper is freely available on [arXiv](http://arxiv.org/abs/1511.06710).

