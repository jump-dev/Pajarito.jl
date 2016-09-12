#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This file implements the default PajaritoSolver
=========================================================#

export PajaritoSolver

immutable PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    log_level::Int              # Verbosity flag: 1 for minimal OA iteration and solve statistics, 2 for including cone summary information, 3 for running commentary
    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    pass_mip_sols::Bool         # Give best feasible solutions constructed from conic subproblem solution to MIP
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    disagg_soc::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
    soc_ell_one::Bool           # (Conic only) Start with disaggregated L_1 outer approximation cuts for SOCs (if disagg_soc)
    soc_ell_inf::Bool           # (Conic only) Start with disaggregated L_inf outer approximation cuts for SOCs (if disagg_soc)
    exp_init::Bool              # (Conic only) Start with several outer approximation cuts on the exponential cones
    drop_dual_infeas::Bool      # (Conic only) Do not add cuts from dual cone infeasible dual vectors
    proj_dual_infeas::Bool      # (Conic only) Project dual cone infeasible dual vectors onto dual cone boundaries
    proj_dual_feas::Bool        # (Conic only) Project dual cone strictly feasible dual vectors onto dual cone boundaries
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver (conic or nonlinear)
    timeout::Float64            # Time limit for outer approximation algorithm not including initial load (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    zero_tol::Float64           # (Conic only) Tolerance for setting small absolute values in duals to zeros
    # sdp_init_lin::Bool          # (Conic SDP only) Use SDP initial linear cuts
    sdp_init_soc::Bool          # (Conic SDP only) Use SDP initial SOC cuts (if MIP solver supports MISOCP)
    sdp_eig::Bool               # (Conic SDP only) Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # (Conic SDP only) Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    sdp_tol_eigvec::Float64     # (Conic SDP only) Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
    sdp_tol_eigval::Float64     # (Conic SDP only) Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues
end


function PajaritoSolver(;
    log_level = 0,
    mip_solver_drives = false,
    pass_mip_sols = true,
    soc_in_mip = false,
    disagg_soc = true,
    soc_ell_one = true,
    soc_ell_inf = true,
    exp_init = true,
    drop_dual_infeas = false,
    proj_dual_infeas = true,
    proj_dual_feas = false,
    mip_solver = MathProgBase.defaultMIPsolver,
    cont_solver = MathProgBase.defaultConicsolver,
    timeout = 60*10,
    rel_gap = 1e-5,
    zero_tol = 1e-10,
    sdp_init_soc = false,
    sdp_eig = true,
    sdp_soc = false,
    sdp_tol_eigvec = 1e-5,
    sdp_tol_eigval = 1e-10
    )

    PajaritoSolver(log_level, mip_solver_drives, pass_mip_sols, soc_in_mip, disagg_soc, soc_ell_one, soc_ell_inf, exp_init, drop_dual_infeas, proj_dual_infeas, proj_dual_feas, mip_solver, cont_solver, timeout, rel_gap, zero_tol, sdp_init_soc, sdp_eig, sdp_soc, sdp_tol_eigvec, sdp_tol_eigval)
end


# Create Pajarito conic model: can solve with either conic algorithm or nonlinear algorithm wrapped with ConicNonlinearBridge
function MathProgBase.ConicModel(s::PajaritoSolver)
    if applicable(MathProgBase.ConicModel, s.cont_solver)
        return PajaritoConicModel(s.log_level, s.mip_solver_drives, s.pass_mip_sols, s.soc_in_mip, s.disagg_soc, s.soc_ell_one, s.soc_ell_inf, s.exp_init, s.drop_dual_infeas, s.proj_dual_infeas, s.proj_dual_feas, s.mip_solver, s.cont_solver, s.timeout, s.rel_gap, s.zero_tol, s.sdp_init_soc, s.sdp_eig, s.sdp_soc, s.sdp_tol_eigvec, s.sdp_tol_eigval)

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

MathProgBase.LinearQuadraticModel(s::PajaritoSolver) = MathProgBase.NonlinearToLPQPBridge(MathProgBase.NonlinearModel(s))
