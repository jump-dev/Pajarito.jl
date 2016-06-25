#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This file implements the default PajaritoSolver
=========================================================#

export PajaritoSolver

immutable PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    path::AbstractString        # File path
    log_level::Int              # Verbosity level flag
    branch_cut::Bool            # Use BC algorithm, else use OA
    misocp::Bool                # Use SOC/SOCRotated cones in the MIP model (if MIP solver supports MISOCP)
    disagg::Bool                # Disaggregate SOC/SOCRotated cones in MIP (if solver is conic)
    drop_dual_infeas::Bool      # Do not add cuts from dual cone infeasible dual vectors
    proj_dual_infeas::Bool      # Project dual cone infeasible dual vectors onto dual cone boundaries
    proj_dual_feas::Bool        # Project dual cone strictly feasible dual vectors onto dual cone boundaries
    solver_mip::MathProgBase.AbstractMathProgSolver # MIP solver
    solver_con::MathProgBase.AbstractMathProgSolver # Continuous solver
    timeout::Float64            # Time limit for OA/BC not including initial load
    tol_rel_opt::Float64        # Relative optimality gap termination condition
    tol_zero::Float64           # Tolerance for setting small values to zeros
    sdp_init_soc::Bool          # Use SDP initial SOC cuts (if MIP solver supports MISOCP)
    sdp_eig::Bool               # Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP branch and cut process)
    sdp_tol_eigvec::Float64     # Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
    sdp_tol_eigval::Float64     # Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues
end


function PajaritoSolver(;
    path = "",
    log_level = 2,
    branch_cut = false,
    misocp = false,
    disagg = true,
    drop_dual_infeas = false,
    proj_dual_infeas = true,
    proj_dual_feas = false,
    solver_mip = MathProgBase.defaultMIPsolver,
    solver_con = MathProgBase.defaultConicsolver,
    timeout = 60*5,
    tol_rel_opt = 1e-5,
    tol_zero = 1e-10,
    sdp_init_soc = true,
    sdp_eig = true,
    sdp_soc = true,
    sdp_tol_eigvec = 1e-5,
    sdp_tol_eigval = 1e-10
    )

    PajaritoSolver(path, log_level, branch_cut, misocp, disagg, drop_dual_infeas, proj_dual_infeas, proj_dual_feas, solver_mip, solver_con, timeout, tol_rel_opt, tol_zero, sdp_init_soc, sdp_eig, sdp_soc, sdp_tol_eigvec, sdp_tol_eigval)
end


# Create Pajarito conic model: can solve with either conic algorithm or nonlinear algorithm wrapped with ConicNonlinearBridge
function MathProgBase.ConicModel(s::PajaritoSolver)
    if applicable(MathProgBase.ConicModel, s.solver_con)
        return PajaritoConicModel(s.path, s.log_level, s.branch_cut, s.misocp, s.disagg, s.drop_dual_infeas, s.proj_dual_infeas, s.proj_dual_feas, s.solver_mip, s.solver_con, s.timeout, s.tol_rel_opt, s.tol_zero, s.sdp_init_soc, s.sdp_eig, s.sdp_soc, s.sdp_tol_eigvec, s.sdp_tol_eigval)

    elseif applicable(MathProgBase.NonlinearModel, s.solver_con)
        return MathProgBase.ConicModel(ConicNonlinearBridge.ConicNLPWrapper(nlp_solver=s))

    else
        error("Continuous solver specified is neither a conic solver nor a nonlinear solver recognized by MathProgBase\n")
    end
end


# Create Pajarito nonlinear model: can solve with nonlinear algorithm only
function MathProgBase.NonlinearModel(s::PajaritoSolver)
    if !applicable(MathProgBase.NonlinearModel, s.solver_con)
        error("Continuous solver specified is not a nonlinear solver recognized by MathProgBase\n")
    end

    # Translate options into old nonlinearmodel.jl fields
    verbose = s.log_level
    algorithm = (s.branch_cut ? "BC" : "OA")
    mip_solver = s.solver_mip
    cont_solver = s.solver_con
    opt_tolerance = s.tol_rel_opt
    time_limit = s.timeout
    instance = s.path

    return PajaritoNonlinearModel(verbose, algorithm, mip_solver, cont_solver, opt_tolerance, time_limit, instance)
end
