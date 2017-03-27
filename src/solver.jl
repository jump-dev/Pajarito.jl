#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This file implements the default PajaritoSolver
=========================================================#

export PajaritoSolver


# Dummy solver
type UnsetSolver <: MathProgBase.AbstractMathProgSolver
end


# Pajarito solver
type PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    log_level::Int              # Verbosity flag: 0 for quiet, 1 for basic solve info, 2 for iteration info, 3 for detailed timing and cuts and solution feasibility info
    timeout::Float64            # Time limit for algorithm (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition

    mip_solver_drives::Bool     # Let MIP solver manage convergence ("branch and cut")
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    mip_subopt_solver::MathProgBase.AbstractMathProgSolver # MIP solver for suboptimal solves (with appropriate options already passed)
    mip_subopt_count::Int       # (Conic only) Number of times to use `mip_subopt_solver` between `mip_solver` solves
    round_mip_sols::Bool        # (Conic only) Round integer variable values before solving subproblems
    use_mip_starts::Bool        # (Conic only) Use conic subproblem feasible solutions as MIP warm-starts or heuristic solutions

    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver (conic or nonlinear)
    solve_relax::Bool           # (Conic only) Solve the continuous conic relaxation to add initial subproblem cuts
    solve_subp::Bool            # (Conic only) Solve the continuous conic subproblems to add subproblem cuts
    dualize_relax::Bool         # (Conic only) Solve the conic dual of the continuous conic relaxation
    dualize_subp::Bool          # (Conic only) Solve the conic duals of the continuous conic subproblems

    soc_disagg::Bool            # (Conic only) Disaggregate SOC cones
    soc_abslift::Bool           # (Conic only) Use SOC absolute value lifting
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP model (if `mip_solver` supports MISOCP)
    sdp_eig::Bool               # (Conic only) Use PSD cone eigenvector cuts
    sdp_soc::Bool               # (Conic only) Use PSD cone eigenvector SOC cuts (if `mip_solver` supports MISOCP)
    init_soc_one::Bool          # (Conic only) Use SOC initial L_1 cuts
    init_soc_inf::Bool          # (Conic only) Use SOC initial L_inf cuts
    init_exp::Bool              # (Conic only) Use Exp initial cuts
    init_sdp_lin::Bool          # (Conic only) Use PSD cone initial linear cuts
    init_sdp_soc::Bool          # (Conic only) Use PSD cone initial SOC cuts (if `mip_solver` supports MISOCP)

    scale_subp_cuts::Bool       # (Conic only) Use scaling for subproblem cuts
    scale_subp_factor::Float64  # (Conic only) Fixed multiplicative factor for scaled subproblem cuts
    viol_cuts_only::Bool        # (Conic only) Only add cuts violated by current MIP solution
    prim_cuts_only::Bool        # (Conic only) Add primal cuts, do not add subproblem cuts
    prim_cuts_always::Bool      # (Conic only) Add primal cuts and subproblem cuts
    prim_cuts_assist::Bool      # (Conic only) Add subproblem cuts, and add primal cuts only subproblem cuts cannot be added

    cut_zero_tol::Float64       # (Conic only) Zero tolerance for cut coefficients
    prim_cut_feas_tol::Float64  # (Conic only) Absolute feasibility tolerance used for primal cuts (set equal to feasibility tolerance of `mip_solver`)
end


function PajaritoSolver(;
    log_level = 1,
    timeout = Inf,
    rel_gap = 1e-5,

    mip_solver_drives = false,
    mip_solver = UnsetSolver(),
    mip_subopt_solver = UnsetSolver(),
    mip_subopt_count = 0,
    round_mip_sols = false,
    use_mip_starts = true,

    cont_solver = UnsetSolver(),
    solve_relax = true,
    solve_subp = true,
    dualize_relax = false,
    dualize_subp = false,

    soc_disagg = true,
    soc_abslift = false,
    soc_in_mip = false,
    sdp_eig = true,
    sdp_soc = false,

    init_soc_one = true,
    init_soc_inf = true,
    init_exp = true,
    init_sdp_lin = true,
    init_sdp_soc = false,

    scale_subp_cuts = true,
    scale_subp_factor = 10.,
    viol_cuts_only = nothing,
    prim_cuts_only = false,
    prim_cuts_always = false,
    prim_cuts_assist = true,

    cut_zero_tol = 1e-10,
    prim_cut_feas_tol = 1e-6,
    )

    if mip_solver == UnsetSolver()
        error("No MIP solver specified (set mip_solver)\n")
    end

    if viol_cuts_only == nothing
        # If user has not set option, default is true on MSD and false on iterative
        viol_cuts_only = mip_solver_drives
    end

    # Deepcopy the solvers because we may change option values inside Pajarito
    PajaritoSolver(log_level, timeout, rel_gap, mip_solver_drives, deepcopy(mip_solver), deepcopy(mip_subopt_solver), mip_subopt_count, round_mip_sols, use_mip_starts, deepcopy(cont_solver), solve_relax, solve_subp, dualize_relax, dualize_subp, soc_disagg, soc_abslift, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, scale_subp_cuts, scale_subp_factor, viol_cuts_only, prim_cuts_only, prim_cuts_always, prim_cuts_assist, cut_zero_tol, prim_cut_feas_tol)
end


# Create Pajarito conic model: can solve with either conic algorithm or nonlinear algorithm wrapped with ConicNonlinearBridge
function MathProgBase.ConicModel(s::PajaritoSolver)
    if applicable(MathProgBase.ConicModel, s.cont_solver) || (s.cont_solver == UnsetSolver())
        if (s.solve_relax || s.solve_subp) && (s.cont_solver == UnsetSolver())
            error("Using conic relaxation (solve_relax) or subproblem solves (solve_subp), but no continuous solver specified (set cont_solver)\n")
        end

        if s.soc_in_mip || s.init_sdp_soc || s.sdp_soc
            # If using MISOCP outer approximation, check MIP solver handles MISOCP
            if !(:SOC in MathProgBase.supportedcones(s.mip_solver))
                error("Using SOC constraints in the MIP model (soc_in_mip or init_sdp_soc or sdp_soc), but MIP solver (mip_solver) specified does not support MISOCP\n")
            end
        end

        if (s.mip_subopt_count > 0) && (s.mip_subopt_solver == UnsetSolver())
            error("Using suboptimal solves (mip_subopt_count > 0), but no suboptimal MIP solver specified (set mip_subopt_solver)\n")
        end

        if s.init_soc_one && !s.soc_disagg && !s.soc_abslift
            error("Cannot use SOC initial L_1 cuts (init_soc_one) if both SOC disaggregation (soc_disagg) and SOC absvalue lifting (soc_abslift) are not used\n")
        end

        if s.sdp_soc && s.mip_solver_drives
            warn("In the MIP-solver-driven algorithm, SOC cuts for SDP cones (sdp_soc) cannot be added from subproblems or primal solutions, but they will be added from the conic relaxation\n")
        end

        if !s.solve_subp
            s.prim_cuts_only = true
            s.use_mip_starts = false
            s.round_mip_sols = false
        end
        if s.prim_cuts_only
            s.prim_cuts_always = true
        end
        if s.prim_cuts_always
            s.prim_cuts_assist = true
        end

        return PajaritoConicModel(s.log_level, s.timeout, s.rel_gap, s.mip_solver_drives, s.mip_solver, s.mip_subopt_solver, s.mip_subopt_count, s.round_mip_sols, s.use_mip_starts, s.cont_solver, s.solve_relax, s.solve_subp, s.dualize_relax, s.dualize_subp, s.soc_disagg, s.soc_abslift, s.soc_in_mip, s.sdp_eig, s.sdp_soc, s.init_soc_one, s.init_soc_inf, s.init_exp, s.init_sdp_lin, s.init_sdp_soc, s.scale_subp_cuts, s.scale_subp_factor, s.viol_cuts_only, s.prim_cuts_only, s.prim_cuts_always, s.prim_cuts_assist, s.cut_zero_tol, s.prim_cut_feas_tol)
    elseif applicable(MathProgBase.NonlinearModel, s.cont_solver)
        return MathProgBase.ConicModel(ConicNonlinearBridge.ConicNLPWrapper(nlp_solver=s))
    else
        error("Continuous solver (cont_solver) specified is not a conic or NLP solver recognized by MathProgBase\n")
    end
end


# Create Pajarito nonlinear model: can solve with nonlinear algorithm only
function MathProgBase.NonlinearModel(s::PajaritoSolver)
    if !applicable(MathProgBase.NonlinearModel, s.cont_solver)
        error("Continuous solver (cont_solver) specified is not a NLP solver recognized by MathProgBase\n")
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


# Return a vector of the supported cone types, for conic algorithm only
function MathProgBase.supportedcones(s::PajaritoSolver)
    if s.cont_solver == UnsetSolver()
        # No conic solver, using primal cuts only, so support all Pajarito cones
        return [:Free, :Zero, :NonNeg, :NonPos, :SOC, :SOCRotated, :SDP, :ExpPrimal]
    elseif applicable(MathProgBase.ConicModel, s.cont_solver)
        # Using conic solver, so supported cones are its cones (plus rotated SOC if SOC is supported)
        cones = MathProgBase.supportedcones(s.cont_solver)
        if :SOC in cones
            push!(cones, :SOCRotated)
        end
        return cones
    else
        # Solver must be NLP
        error("Cannot get cones supported by continuous solver (cont_solver)\n")
    end
end
