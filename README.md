# Pajarito

Pajarito (**P**olyhedral **A**pproximation in **J**ulia: **A**utomatic **R**eformulations for **I**n**T**eger **O**ptimization) is a mixed-integer convex programming solver package written in [Julia](http://julialang.org/). Pajarito combines state-of-the-art convex and mixed-integer linear solvers by constructing sequential polyhedral approximations of the convex parts with a guaranteed finite-time convergence under minimal assumptions. Pajarito supports both **mixed-integer conic programming** and **smooth, derivative-based mixed-integer nonlinear programming**.

[![Build Status](https://travis-ci.org/mlubin/Pajarito.jl.svg?branch=master)](https://travis-ci.org/mlubin/Pajarito.jl) [![codecov.io](https://codecov.io/github/mlubin/Pajarito.jl/coverage.svg?branch=master)](https://codecov.io/github/mlubin/Pajarito.jl?branch=master) [![Pajarito](http://pkg.julialang.org/badges/Pajarito_0.5.svg)](http://pkg.julialang.org/?pkg=Pajarito&ver=0.5)

## Installation

Pajarito can be installed through the Julia package manager:

```
julia> Pkg.add("Pajarito")
```

## Should I use Pajarito?

Pajarito is a solver for mixed-integer convex optimization problems. These are optimization problems where some variables are restricted to take discrete (e.g. binary) values, and the problem becomes convex when the discreteness constraints are relaxed. If your problem falls into this class, you may consider using Pajarito.

## Using Pajarito

Pajarito has two entirely separate algorithms depending on the form in which the input is provided. The first is the **derivative-based nonlinear** algorithm, where the approach used is analogous to that of [Bonmin](https://projects.coin-or.org/Bonmin) and will perform similarly, with the primary advantage of being able to easily swap-in various mixed-integer and convex subproblem solvers which Bonmin does not support. The second is **conic** algorithm which is the new approach proposed in the publications describing Pajarito. The conic algorithm currently supports second-order cones, exponential cones, and positive semidefinite cones. A problem provided in conic form may solve faster than an identical problem encoded in derivative-based nonlinear form because conic form naturally encodes extended formulations; however, [Hijazi et al.](http://www.optimization-online.org/DB_FILE/2011/06/3050.pdf) suggest manual reformulation techniques which achieve many of the algorithmic benefits of conic form.

The table below describes the different ways to access the two algorithms in Pajarito.

|                             | [JuMP][JuMP-url]            | [Convex.jl][convex-url]  | [MathProgBase][mpb-url] |
|-----------------------------|-----------------------------|--------------------------|-------------------------|
| Derivative-based nonlinear  | X                           |                          | [X][mpb-nlp-url]        |
| Conic (incl. MISOCP, MISDP) | X (no automatic conversion) | X (automatic conversion) | [X][mpb-conic-url]      |

* MISOCP: mixed-integer second-order cone programming
* MISDP: mixed-integer semidefinite programming

JuMP and Convex.jl are algebraic modeling interfaces, while MathProgBase is a lower-level interface for providing input in raw callback or matrix form.

[mpb-nlp-url]: http://mathprogbasejl.readthedocs.io/en/latest/nlp.html
[mpb-conic-url]: http://mathprogbasejl.readthedocs.io/en/latest/conic.html
[JuMP-url]: https://github.com/JuliaOpt/JuMP.jl
[convex-url]: https://github.com/JuliaOpt/Convex.jl
[mpb-url]: https://github.com/JuliaOpt/MathProgBase.jl

Pajarito may be accessed through MathProgBase from outside Julia by using the experimental [cmpb](https://github.com/mlubin/cmpb) interface which provides a C API to the low-level conic input format.

## Subproblem solvers

The algorithm implemented by Pajarito itself is relatively simple, and most of the hard work is performed by subproblem solvers. Pajarito requires two different subproblem solvers, one for mixed-integer linear problems and one for convex subproblems. **The performance of Pajarito depends on the subproblem solvers.** For best performance, use commercial solvers.

The mixed-integer linear solver is specified by using the `mip_solver` option to `PajaritoSolver`, e.g., `PajaritoSolver(mip_solver=CplexSolver())`. You must first load the Julia package which provides the mixed-integer linear solver, e.g., with `using CPLEX`.

The convex subproblem solver is specified by using the `cont_solver` option, e.g., `PajaritoSolver(cont_solver=IpoptSolver())`. When given input in derivative-based nonlinear form, Pajarito requires a derivative-based nonlinear solver, e.g., [Ipopt](https://projects.coin-or.org/Ipopt) or [KNITRO](http://www.ziena.com/knitro.htm). When given input in conic form, the convex subproblem solver can be *either* a conic solver like [ECOS](https://github.com/JuliaOpt/ECOS.jl) *or* a derivative-based solver like Ipopt. If a derivative-based solver is provided in this case, then Pajarito will switch to the derivative-based algorithm by using the [ConicNonlinearBridge](https://github.com/mlubin/ConicNonlinearBridge.jl) package. Note that using derivative-based solvers for conic problems can cause numerical instability because conic problems are not always smooth.

All solvers can have their parameters specified through their corresponding Julia interfaces. For example, you probably should turn off the output of the subproblem solvers, e.g., by using `IpoptSolver(print_level=0)`.

## Pajarito solver options

The following options can be passed to `PajaritoSolver()` to modify its behavior (**C** means conic algorithm only):

  * `log_level::Int` Verbosity flag: -1 for no output, 0 for minimal solution information, 1 for basic OA iteration and solve statistics, 2 for cone summary information, 3 for infeasibilities of duals, cuts, and OA solutions
  * `timeout::Float64` Time limit for outer approximation algorithm not including initial load (in seconds)
  * `rel_gap::Float64` Relative optimality gap termination condition
  * `mip_solver_drives::Bool` Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
  * `mip_solver::MathProgBase.AbstractMathProgSolver` MIP solver (MILP or MISOCP)
  * `mip_subopt_solver::MathProgBase.AbstractMathProgSolver` **C** MIP solver for suboptimal solves, with appropriate options (gap or timeout) specified directly
  * `mip_subopt_count::Int` **C** Number of times to solve MIP suboptimally with time limit between zero gap solves
  * `round_mip_sols::Bool` **C** Round the integer variable values from the MIP solver before passing to the conic subproblems
  * `pass_mip_sols::Bool` **C** Give best feasible solutions constructed from conic subproblem solution to MIP
  * `cont_solver::MathProgBase.AbstractMathProgSolver` Continuous solver (conic or nonlinear)
  * `solve_relax::Bool` **C** Solve the continuous conic relaxation to add initial dual cuts
  * `dualize_relax::Bool` **C** Solve the conic dual of the continuous conic relaxation
  * `dualize_sub::Bool` **C** Solve the conic duals of the continuous conic subproblems
  * `soc_disagg::Bool` **C** Disaggregate SOC cones in the MIP only
  * `soc_in_mip::Bool` **C** Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
  * `sdp_eig::Bool` **C** Use SDP eigenvector-derived cuts
  * `sdp_soc::Bool` **C** Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
  * `init_soc_one::Bool` **C** Start with disaggregated L_1 outer approximation cuts for SOCs (if soc_disagg)
  * `init_soc_inf::Bool` **C** Start with disaggregated L_inf outer approximation cuts for SOCs (if soc_disagg)
  * `init_exp::Bool` **C** Start with several outer approximation cuts on the exponential cones
  * `init_sdp_lin::Bool` **C** Use SDP initial linear cuts
  * `init_sdp_soc::Bool` **C** Use SDP initial SOC cuts (if MIP solver supports MISOCP)
  * `viol_cuts_only::Bool` **C** Only add cuts that are violated by the current MIP solution (may be useful for MSD algorithm where many cuts are added)
  * `proj_dual_infeas::Bool` **C** Project dual cone infeasible dual vectors onto dual cone boundaries
  * `proj_dual_feas::Bool` **C** Project dual cone strictly feasible dual vectors onto dual cone boundaries
  * `prim_cuts_only::Bool` **C** Do not add dual cuts
  * `prim_cuts_always::Bool` **C** Add primal cuts at each iteration or in each lazy callback
  * `prim_cuts_assist::Bool` **C** Add primal cuts only when integer solutions are repeating
  * `tol_zero::Float64` **C** Tolerance for setting small absolute values in duals to zeros
  * `tol_prim_zero::Float64` **C** Tolerance level for zeros in primal cut adding functions (must be at least 1e-5)
  * `tol_prim_infeas::Float64` **C** Tolerance level for cone outer infeasibilities for primal cut adding functions (must be at least 1e-5)
  * `tol_sdp_eigvec::Float64` **C** Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
  * `tol_sdp_eigval::Float64` **C** Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues

**Pajarito is not yet numerically robust and may require tuning of parameters to improve convergence.** If the default parameters don't work for you, please let us know.

## Bug reports and support

Please report any issues via the Github **[issue tracker]**. All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc. The **[Optimization (Mathematical)]** category on Discourse is appropriate for general discussion.

[issue tracker]: https://github.com/mlubin/Pajarito.jl/issues
[Optimization (Mathematical)]: https://discourse.julialang.org/c/domain/opt

## We need your challenging MICP problems

Mixed-integer convex programming is an active area of research, and we are seeking out hard benchmark instances. Please get in touch either by opening an issue or privately if you would like to share any hard instances to be used as benchmarks in future work. Challenging problems will help us determine how to improve Pajarito.

## References

If you find Pajarito useful in your work, we kindly request that you cite the following [paper](http://dx.doi.org/10.1007/978-3-319-33461-5_9) ([arXiv preprint](http://arxiv.org/abs/1511.06710)):

    @Inbook{LubinYamangilBentVielma2016,
    author="Lubin, Miles
    and Yamangil, Emre
    and Bent, Russell
    and Vielma, Juan Pablo",
    editor="Louveaux, Quentin
    and Skutella, Martin",
    title="Extended Formulations in Mixed-Integer Convex Programming",
    bookTitle="Integer Programming and Combinatorial Optimization: 18th International Conference, IPCO 2016, Li{\`e}ge, Belgium, June 1-3, 2016, Proceedings",
    year="2016",
    publisher="Springer International Publishing",
    address="Cham",
    pages="102--113",
    isbn="978-3-319-33461-5",
    doi="10.1007/978-3-319-33461-5_9",
    url="http://dx.doi.org/10.1007/978-3-319-33461-5_9"
    }

The paper describes the motivation of Pajarito and is recommended reading for advanced users.
