#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
 Pajarito solver object
=========================================================#

export PajaritoSolver

# Dummy solver
struct UnsetSolver <: MathProgBase.AbstractMathProgSolver end

# Pajarito solver
mutable struct PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    log_level::Int              # Verbosity flag: 0 for quiet, 1 for basic solve info, 2 for iteration info, 3 for detailed timing and cuts and solution feasibility info
    timeout::Float64            # Time limit for algorithm (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition

    mip_solver_drives::Bool     # Let MIP solver manage convergence ("branch and cut")
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    mip_subopt_solver::MathProgBase.AbstractMathProgSolver # MIP solver for suboptimal solves (with appropriate options already passed)
    mip_subopt_count::Int       # Number of times to use `mip_subopt_solver` between `mip_solver` solves
    round_mip_sols::Bool        # Round integer variable values before solving subproblems
    use_mip_starts::Bool        # Use conic subproblem feasible solutions as MIP warm-starts or heuristic solutions

    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous conic solver
    solve_relax::Bool           # Solve the continuous conic relaxation to add initial subproblem cuts
    solve_subp::Bool            # Solve the continuous conic subproblems to add subproblem cuts
    dualize_relax::Bool         # Solve the conic dual of the continuous conic relaxation
    dualize_subp::Bool          # Solve the conic duals of the continuous conic subproblems

    all_disagg::Bool            # Disaggregate cuts on the nonpolyhedral cones
    soc_disagg::Bool            # Disaggregate SOC cones
    soc_abslift::Bool           # Use SOC absolute value lifting
    soc_in_mip::Bool            # Use SOC cones in the MIP model (if `mip_solver` supports MISOCP)
    sdp_eig::Bool               # Use PSD cone eigenvector cuts
    sdp_soc::Bool               # Use PSD cone eigenvector SOC cuts (if `mip_solver` supports MISOCP)
    init_soc_one::Bool          # Use SOC initial L_1 cuts
    init_soc_inf::Bool          # Use SOC initial L_inf cuts
    init_exp::Bool              # Use Exp initial cuts
    init_sdp_lin::Bool          # Use PSD cone initial linear cuts
    init_sdp_soc::Bool          # Use PSD cone initial SOC cuts (if `mip_solver` supports MISOCP)

    scale_subp_cuts::Bool       # Use scaling for subproblem cuts
    scale_subp_factor::Float64  # Fixed multiplicative factor for scaled subproblem cuts
    scale_subp_up::Bool         # Scale up any scaled subproblem cuts that are smaller than the equivalent separation cut
    viol_cuts_only::Bool        # Only add cuts violated by current MIP solution
    prim_cuts_only::Bool        # Add primal cuts, do not add subproblem cuts
    prim_cuts_always::Bool      # Add primal cuts and subproblem cuts
    prim_cuts_assist::Bool      # Add subproblem cuts, and add primal cuts only subproblem cuts cannot be added

    cut_zero_tol::Float64       # Zero tolerance for cut coefficients
    prim_cut_feas_tol::Float64  # Absolute feasibility tolerance used for primal cuts (set equal to feasibility tolerance of `mip_solver`)

    dump_subproblems::Bool      # Save each conic subproblem in conic benchmark format (CBF) at a specified location
    dump_basename::String       # Basename of conic subproblem CBF files: "/path/to/foo" creates files "/path/to/foo_NN.cbf" where "NN" is a counter
end

function PajaritoSolver(;
    log_level = 1,
    timeout = Inf,
    rel_gap = 1e-5,

    mip_solver_drives = nothing,
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

    all_disagg = true,
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
    scale_subp_up = false,
    viol_cuts_only = nothing,
    prim_cuts_only = false,
    prim_cuts_always = false,
    prim_cuts_assist = true,

    cut_zero_tol = 1e-12,
    prim_cut_feas_tol = 1e-6,

    dump_subproblems = false,
    dump_basename = nothing,
    )

    if (cont_solver != UnsetSolver()) && !applicable(MathProgBase.ConicModel, cont_solver)
        error("Continuous solver (cont_solver) specified is not a conic solver; if your continuous solver is a derivative-based NLP solver, try Pavito solver (Pajarito's MINLP functionality was moved to the Pavito solver package)\n")
    end

    if mip_solver == UnsetSolver()
        error("No MIP solver specified (set mip_solver)\n")
    end
    if mip_solver_drives == nothing
        mip_solver_drives = applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip_solver), x -> x)
    elseif mip_solver_drives && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip_solver), x -> x)
        error("MIP solver does not support callbacks (cannot set mip_solver_drives = true)")
    end

    if viol_cuts_only == nothing
        # If user has not set option, default is true on MSD and false on iterative
        viol_cuts_only = mip_solver_drives
    end

    if dump_basename == nothing
        if dump_subproblems
            error("No basename set for conic subproblem dumps (set dump_basename)\n")
        else
            dump_basename = ""
        end
    end

    # Deepcopy the solvers because we may change option values inside Pajarito
    PajaritoSolver(log_level, timeout, rel_gap, mip_solver_drives, deepcopy(mip_solver), deepcopy(mip_subopt_solver), mip_subopt_count, round_mip_sols, use_mip_starts, deepcopy(cont_solver), solve_relax, solve_subp, dualize_relax, dualize_subp, all_disagg, soc_disagg, soc_abslift, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, scale_subp_cuts, scale_subp_factor, scale_subp_up, viol_cuts_only, prim_cuts_only, prim_cuts_always, prim_cuts_assist, cut_zero_tol, prim_cut_feas_tol, dump_subproblems, dump_basename)
end

# Cannot use Pajarito on an NLP model
MathProgBase.NonlinearModel(s::PajaritoSolver) = error("Pajarito solver cannot be used for NLP models (Pajarito's MINLP functionality was moved to the Pavito solver package)\n")

# Create Pajarito conic model
function MathProgBase.ConicModel(s::PajaritoSolver)
    if (s.solve_relax || s.solve_subp) && (s.cont_solver == UnsetSolver())
        error("Using conic relaxation (solve_relax) or subproblem solves (solve_subp), but no continuous solver specified (set cont_solver)\n")
    end

    if s.soc_in_mip || s.init_sdp_soc || s.sdp_soc
        # If using MISOCP outer approximation, check MIP solver handles MISOCP
        if !(:SOC in MathProgBase.supportedcones(s.mip_solver))
            error("Using SOC constraints in the MIP model (soc_in_mip or init_sdp_soc or sdp_soc), but MIP solver (mip_solver) specified does not support MISOCP\n")
        end
    end

    if !s.all_disagg && (s.soc_disagg || s.sdp_eig || s.sdp_soc)
        error("Cannot use SOC extended formulation (soc_disagg) or SDP cut disaggregation (sdp_eig) or SOC cuts for SDP cones (sdp_soc) when not also disaggregating all nonpolyhedral cone cuts (all_disagg)\n")
    end

    if (s.mip_subopt_count > 0) && (s.mip_subopt_solver == UnsetSolver())
        error("Using suboptimal solves (mip_subopt_count > 0), but no suboptimal MIP solver specified (set mip_subopt_solver)\n")
    end

    if s.init_soc_one && !s.soc_disagg && !s.soc_abslift
        error("Cannot use SOC initial L_1 cuts (init_soc_one) if both SOC disaggregation (soc_disagg) and SOC absvalue lifting (soc_abslift) are not used\n")
    end

    if s.sdp_soc && s.mip_solver_drives
        @warn "In the MIP-solver-driven algorithm, SOC cuts for SDP cones (sdp_soc) cannot be added from subproblems or primal solutions, but they will be added from the conic relaxation\n"
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

    if !s.all_disagg && s.prim_cuts_assist
        error("Cannot use primal cuts when not disaggregating all nonpolyhedral cone cuts (all_disagg)\n")
    end

    return PajaritoConicModel(s.log_level, s.timeout, s.rel_gap, s.mip_solver_drives, s.mip_solver, s.mip_subopt_solver, s.mip_subopt_count, s.round_mip_sols, s.use_mip_starts, s.cont_solver, s.solve_relax, s.solve_subp, s.dualize_relax, s.dualize_subp, s.all_disagg, s.soc_disagg, s.soc_abslift, s.soc_in_mip, s.sdp_eig, s.sdp_soc, s.init_soc_one, s.init_soc_inf, s.init_exp, s.init_sdp_lin, s.init_sdp_soc, s.scale_subp_cuts, s.scale_subp_factor, s.scale_subp_up, s.viol_cuts_only, s.prim_cuts_only, s.prim_cuts_always, s.prim_cuts_assist, s.cut_zero_tol, s.prim_cut_feas_tol, s.dump_subproblems, s.dump_basename)
end

# Create Pajaito LinearQuadratic model
MathProgBase.LinearQuadraticModel(s::PajaritoSolver) = MathProgBase.ConicToLPQPBridge(MathProgBase.ConicModel(s))

# Return a vector of the supported cone types
function MathProgBase.supportedcones(s::PajaritoSolver)
    if s.cont_solver == UnsetSolver()
        # No conic solver, using primal cuts only, so support all Pajarito cones
        return [:Free, :Zero, :NonNeg, :NonPos, :SOC, :SOCRotated, :SDP, :ExpPrimal]
    else
        # Using conic solver, so supported cones are its cones (plus rotated SOC if SOC is supported)
        cones = MathProgBase.supportedcones(s.cont_solver)
        if :SOC in cones
            push!(cones, :SOCRotated)
        end
        return cones
    end
end
