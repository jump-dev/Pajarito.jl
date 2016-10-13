#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This file implements the default PajaritoSolver
=========================================================#

export PajaritoSolver

immutable PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    # Solver parameters
    log_level::Int              # Verbosity flag: 1 for minimal OA iteration and solve statistics, 2 for including cone summary information, 3 for running commentary
    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    pass_mip_sols::Bool         # (Conic only) Give best feasible solutions constructed from conic subproblem solution to MIP
    round_mip_sols::Bool        # (Conic only) Round the integer variable values from the MIP solver before passing to the conic subproblems
    mip_subopt_count::Int       # (Conic only) Number of times to solve MIP suboptimally with time limit between zero gap solves
    mip_subopt_solver::MathProgBase.AbstractMathProgSolver # MIP solver for suboptimal solves, with appropriate options (gap or timeout) specified directly
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    disagg_soc::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
    soc_ell_one::Bool           # (Conic only) Start with disaggregated L_1 outer approximation cuts for SOCs (if disagg_soc)
    soc_ell_inf::Bool           # (Conic only) Start with disaggregated L_inf outer approximation cuts for SOCs (if disagg_soc)
    exp_init::Bool              # (Conic only) Start with several outer approximation cuts on the exponential cones
    proj_dual_infeas::Bool      # (Conic only) Project dual cone infeasible dual vectors onto dual cone boundaries
    proj_dual_feas::Bool        # (Conic only) Project dual cone strictly feasible dual vectors onto dual cone boundaries
    viol_cuts_only::Bool        # (Conic only) Only add cuts that are violated by the current MIP solution (may be useful for MSD algorithm where many cuts are added)
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver (conic or nonlinear)
    timeout::Float64            # Time limit for outer approximation algorithm not including initial load (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    detect_slacks::Bool         # (Conic only) Use automatic slack variable detection for cuts (may reduce number of variables in MIP)
    slack_tol_order::Float64    # (Conic only) Order of magnitude tolerance for abs of coefficient on auto-detected slack variables (negative: -1 only, zero: -1 or 1, positive: order of magnitude)
    zero_tol::Float64           # (Conic only) Tolerance for setting small absolute values in duals to zeros
    primal_cuts_only::Bool      # (Conic only) Do not add dual cuts
    primal_cuts_always::Bool    # (Conic only) Add primal cuts at each iteration or in each lazy callback
    primal_cuts_assist::Bool    # (Conic only) Add primal cuts only when integer solutions are repeating
    primal_cut_zero_tol::Float64 # (Conic only) Tolerance level for zeros in primal cut adding functions (must be at least 1e-5)
    primal_cut_inf_tol::Float64 # (Conic only) Tolerance level for cone outer infeasibilitities for primal cut adding functions (must be at least 1e-5)
    sdp_init_lin::Bool          # (Conic SDP only) Use SDP initial linear cuts
    sdp_init_soc::Bool          # (Conic SDP only) Use SDP initial SOC cuts (if MIP solver supports MISOCP)
    sdp_eig::Bool               # (Conic SDP only) Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # (Conic SDP only) Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    sdp_tol_eigvec::Float64     # (Conic SDP only) Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
    sdp_tol_eigval::Float64     # (Conic SDP only) Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues
end


function PajaritoSolver(;
    log_level = 1,
    mip_solver_drives = false,
    pass_mip_sols = true,
    round_mip_sols = false,
    mip_subopt_count = 0,
    mip_subopt_solver = MathProgBase.defaultMIPsolver,
    soc_in_mip = false,
    disagg_soc = true,
    soc_ell_one = true,
    soc_ell_inf = true,
    exp_init = true,
    proj_dual_infeas = true,
    proj_dual_feas = false,
    viol_cuts_only = false,
    mip_solver = MathProgBase.defaultMIPsolver,
    cont_solver = MathProgBase.defaultConicsolver,
    timeout = 60*10.,
    rel_gap = 1e-5,
    detect_slacks = true,
    slack_tol_order = 2.,
    zero_tol = 1e-10,
    primal_cuts_only = false,
    primal_cuts_always = false,
    primal_cuts_assist = false,
    primal_cut_zero_tol = 1e-4,
    primal_cut_inf_tol = 1e-6,
    sdp_init_lin = true,
    sdp_init_soc = false,
    sdp_eig = true,
    sdp_soc = false,
    sdp_tol_eigvec = 1e-2,
    sdp_tol_eigval = 1e-6
    )

    PajaritoSolver(log_level, mip_solver_drives, pass_mip_sols, round_mip_sols, mip_subopt_count, mip_subopt_solver, soc_in_mip, disagg_soc, soc_ell_one, soc_ell_inf, exp_init, proj_dual_infeas, proj_dual_feas, viol_cuts_only, mip_solver, cont_solver, timeout, rel_gap, detect_slacks, slack_tol_order, zero_tol, primal_cuts_only, primal_cuts_always, primal_cuts_assist, primal_cut_zero_tol, primal_cut_inf_tol, sdp_init_lin, sdp_init_soc, sdp_eig, sdp_soc, sdp_tol_eigvec, sdp_tol_eigval)
end


# Create Pajarito conic model: can solve with either conic algorithm or nonlinear algorithm wrapped with ConicNonlinearBridge
function MathProgBase.ConicModel(s::PajaritoSolver)
    if applicable(MathProgBase.ConicModel, s.cont_solver)
        return PajaritoConicModel(s.log_level, s.mip_solver_drives, s.pass_mip_sols, s.round_mip_sols, s.mip_subopt_count, s.mip_subopt_solver, s.soc_in_mip, s.disagg_soc, s.soc_ell_one, s.soc_ell_inf, s.exp_init, s.proj_dual_infeas, s.proj_dual_feas, s.viol_cuts_only, s.mip_solver, s.cont_solver, s.timeout, s.rel_gap, s.detect_slacks, s.slack_tol_order, s.zero_tol, s.primal_cuts_only, s.primal_cuts_always, s.primal_cuts_assist, s.primal_cut_zero_tol, s.primal_cut_inf_tol, s.sdp_init_lin, s.sdp_init_soc, s.sdp_eig, s.sdp_soc, s.sdp_tol_eigvec, s.sdp_tol_eigval)
    elseif applicable(MathProgBase.NonlinearModel, s.cont_solver)
        return MathProgBase.ConicModel(ConicNonlinearBridge.ConicNLPWrapper(nlp_solver=s))
    else
        error("Continuous solver specified is neither a conic solver nor a nonlinear solver recognized by MathProgBase\n")
    end
end


# Create Pajarito nonlinear model: can solve with nonlinear algorithm only
function MathProgBase.NonlinearModel(s::PajaritoSolver)
    if !applicable(MathProgBase.NonlinearModel, s.cont_solver)
        error("Continuous solver specified is not a nonlinear solver recognized by MathProgBase\n")
    end

    # Translate options into old nonlinearmodel.jl fields
    verbose = s.log_level
    algorithm = (s.mip_solver_drives ? "BC" : "OA")
    mip_solver = s.mip_solver
    cont_solver = s.cont_solver
    opt_tolerance = s.rel_gap
    time_limit = s.timeout

    return PajaritoNonlinearModel(verbose, algorithm, mip_solver, cont_solver, opt_tolerance, time_limit)
end


# Create Pajarito linear-quadratic model: can solve with nonlinear algorithm wrapped with NonlinearToLPQPBridge
MathProgBase.LinearQuadraticModel(s::PajaritoSolver) = MathProgBase.NonlinearToLPQPBridge(MathProgBase.NonlinearModel(s))
