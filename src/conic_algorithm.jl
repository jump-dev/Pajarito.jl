#  Copyright 2016, Los Alamos National Laboratory, LANS LLC, and Chris Coey.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, you can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This mixed-integer conic programming algorithm is described in:
  Lubin, Yamangil, Bent, Vielma (2016), Extended formulations
  in Mixed-Integer Convex Programming, IPCO 2016, Liege, Belgium
  (available online at http://arxiv.org/abs/1511.06710)

Model MICP with JuMP.jl conic format or Convex.jl DCP format
http://mathprogbasejl.readthedocs.org/en/latest/conic.html


TODO features
- implement warm-starting: use set_best_soln!
- enable querying logs information etc

=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
    log_level::Int              # Verbosity flag: -1 for no output, 0 for minimal solution information, 1 for basic OA iteration and solve statistics, 2 for cone summary information, 3 for infeasibilities of duals, cuts, and OA solutions
    timeout::Float64            # Time limit for outer approximation algorithm not including initial load (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition

    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver (MILP or MISOCP)
    mip_subopt_solver::MathProgBase.AbstractMathProgSolver # MIP solver for suboptimal solves, with appropriate options (gap or timeout) specified directly
    mip_subopt_count::Int       # (Conic only) Number of times to solve MIP suboptimally with time limit between zero gap solves
    round_mip_sols::Bool        # (Conic only) Round the integer variable values from the MIP solver before passing to the conic subproblems
    pass_mip_sols::Bool         # (Conic only) Give best feasible solutions constructed from conic subproblem solution to MIP

    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver (conic or nonlinear)
    solve_relax::Bool           # (Conic only) Solve the continuous conic relaxation to add initial subproblem cuts
    dualize_relax::Bool         # (Conic only) Solve the conic dual of the continuous conic relaxation
    dualize_sub::Bool           # (Conic only) Solve the conic duals of the continuous conic subproblems

    soc_disagg::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
    soc_abslift::Bool           # (Conic only) Use absolute value lifting in the MIP only
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    sdp_eig::Bool               # (Conic SDP only) Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # (Conic SDP only) Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    init_soc_one::Bool          # (Conic only) Start with disaggregated L_1 outer approximation cuts for SOCs
    init_soc_inf::Bool          # (Conic only) Start with disaggregated L_inf outer approximation cuts for SOCs
    init_exp::Bool              # (Conic Exp only) Start with several outer approximation cuts on the exponential cones
    init_sdp_lin::Bool          # (Conic SDP only) Use SDP initial linear cuts
    init_sdp_soc::Bool          # (Conic SDP only) Use SDP initial SOC cuts (if MIP solver supports MISOCP)

    scale_subp_cuts::Bool       # (Conic only) Use scaling for subproblem cuts based on subproblem status
    viol_cuts_only::Bool        # (Conic only) Only add cuts that are violated by the current MIP solution (may be useful for MSD algorithm where many cuts are added)
    prim_cuts_only::Bool        # (Conic only) Do not add subproblem cuts
    prim_cuts_always::Bool      # (Conic only) Add primal cuts at each iteration or in each lazy callback
    prim_cuts_assist::Bool      # (Conic only) Add primal cuts only when integer solutions are repeating

    tol_zero::Float64           # (Conic only) Tolerance for small epsilons as zeros
    tol_prim_infeas::Float64    # (Conic only) Tolerance level for cone outer infeasibilities for primal cut adding functions (must be at least 1e-5)

    # Initial data
    num_var_orig::Int           # Initial number of variables
    num_con_orig::Int           # Initial number of constraints
    c_orig                      # Initial objective coefficients vector
    A_orig                      # Initial affine constraint matrix (sparse representation)
    b_orig                      # Initial constraint right hand side
    cone_con_orig               # Initial constraint cones vector (cone, index)
    cone_var_orig               # Initial variable cones vector (cone, index)
    var_types::Vector{Symbol}   # Variable types vector on original variables (only :Bin, :Cont, :Int)
    # var_start::Vector{Float64}  # Variable warm start vector on original variables

    # Conic subproblem data
    cone_con_sub::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data in conic subproblem
    cone_var_sub::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data in conic subproblem
    A_sub_cont::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    A_sub_int::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and integer variable columns
    b_sub::Vector{Float64}      # Subvector of b containing full rows
    c_sub_cont::Vector{Float64} # Subvector of c for continuous variables
    c_sub_int::Vector{Float64}  # Subvector of c for integer variables

    # MIP data
    model_mip::JuMP.Model       # JuMP MIP (outer approximation) model
    x_int::Vector{JuMP.Variable} # JuMP (sub)vector of integer variables
    x_cont::Vector{JuMP.Variable} # JuMP (sub)vector of continuous variables

    # SOC data
    num_soc::Int                # Number of SOCs
    t_idx_soc_subp::Vector{Int} # Row index of t variable in SOCs in subproblems
    v_idxs_soc_subp::Vector{Vector{Int}} # Row indices of v variables in SOCs in subproblem
    t_soc::Vector{JuMP.AffExpr} # t variable (epigraph) in SOCs
    v_soc::Vector{Vector{JuMP.AffExpr}} # v variables in SOCs
    d_soc::Vector{Vector{JuMP.Variable}} # d variables (disaggregated) in SOCs
    a_soc::Vector{Vector{JuMP.Variable}} # a variables (absolute values) in SOCs

    # ExpPrimal data
    num_exp::Int                # Number of ExpPrimal cones
    r_idx_exp_subp::Vector{Int}
    s_idx_exp_subp::Vector{Int}
    t_idx_exp_subp::Vector{Int}
    r_exp::Vector{JuMP.AffExpr}
    s_exp::Vector{JuMP.AffExpr}
    t_exp::Vector{JuMP.AffExpr}

    # SDP data
    num_sdp::Int                # Number of SDP cones
    # rows_sub_sdp::Vector{Vector{Int}} # Row indices in subproblem
    # dim_sdp::Vector{Int}        # Dimensions
    # # vars_svec_sdp::Vector{Vector{JuMP.Variable}} # Slack variables in svec form (newly added or detected)
    # vars_sdp::Vector{Array{JuMP.AffExpr,2}} # Slack variables in smat form (newly added or detected)
    # smat::Vector{Array{Float64,2}} # Preallocated matrix to help with memory for SDP cut generation

    # Miscellaneous for algorithms
    update_conicsub::Bool       # Indicates whether to use setbvec! to update an existing conic subproblem model
    model_conic::MathProgBase.AbstractConicModel # Conic subproblem model: persists when the conic solver implements MathProgBase.setbvec!
    oa_started::Bool            # Indicator for Iterative or MIP-solver-driven algorithms started
    cache_soln::Set{Vector{Float64}} # Set of integer solution subvectors already seen
    new_incumb::Bool            # Indicates whether a new incumbent solution from the conic solver is waiting to be added as warm-start or heuristic
    cb_heur                     # Heuristic callback reference (MIP-driven only)
    cb_lazy                     # Lazy callback reference (MIP-driven only)

    # Solution and bound information
    mip_obj::Float64            # Latest MIP (outer approx) objective value
    best_obj::Float64           # Best feasible objective value
    best_int::Vector{Float64}   # Best feasible integer solution
    best_cont::Vector{Float64}  # Best feasible continuous solution
    best_slck::Vector{Float64}  # Best feasible slack vector (for calculating MIP solution)
    gap_rel_opt::Float64        # Relative optimality gap = |mip_obj - best_obj|/|best_obj|
    final_soln::Vector{Float64} # Final solution on original variables

    # Logging information and status
    logs::Dict{Symbol,Any}
    status::Symbol

    # Model constructor
    function PajaritoConicModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, mip_subopt_solver, mip_subopt_count, round_mip_sols, pass_mip_sols, cont_solver, solve_relax, dualize_relax, dualize_sub, soc_disagg, soc_abslift, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, scale_subp_cuts, viol_cuts_only, prim_cuts_only, prim_cuts_always, prim_cuts_assist, tol_zero, tol_prim_infeas)
        # Errors
        if soc_in_mip || init_sdp_soc || sdp_soc
            # If using MISOCP outer approximation, check MIP solver handles MISOCP
            mip_spec = MathProgBase.supportedcones(mip_solver)
            if !(:SOC in mip_spec)
                error("The MIP solver specified does not support MISOCP\n")
            end
        end
        if prim_cuts_only && !prim_cuts_always
            error("When using primal cuts only, they are also added always (set prim_cuts_always = prim_cuts_assist = true)\n")
        end
        if prim_cuts_always && !prim_cuts_assist
            error("When using primal cuts always, they are also added for assistance (set prim_cuts_assist = true)\n")
        end
        if init_soc_one && !(soc_disagg || soc_abslift)
            error("Cannot use initial SOC L_1 constraints if not using SOC disaggregation or SOC absvalue lifting\n")
        end

        # Warnings
        if log_level > 1
            if !solve_relax
                warn("Not solving the conic continuous relaxation problem; Pajarito may fail if the outer approximation MIP is unbounded\n")
            end
            if sdp_soc && mip_solver_drives
                warn("SOC cuts for SDP cones cannot be added during the MIP-solver-driven algorithm, but initial SOC cuts may be used\n")
            end
            if round_mip_sols
                warn("Integer solutions will be rounded: if this seems to cause numerical challenges, change round_mip_sols option\n")
            end
            if prim_cuts_only
                warn("Using primal cuts only may cause convergence issues\n")
            end
        end

        # Initialize model
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.solve_relax = solve_relax
        m.dualize_relax = dualize_relax
        m.dualize_sub = dualize_sub
        m.pass_mip_sols = pass_mip_sols
        m.round_mip_sols = round_mip_sols
        m.mip_subopt_count = mip_subopt_count
        m.mip_subopt_solver = mip_subopt_solver
        m.soc_in_mip = soc_in_mip
        m.soc_disagg = soc_disagg
        m.soc_abslift = soc_abslift
        m.init_soc_one = init_soc_one
        m.init_soc_inf = init_soc_inf
        m.init_exp = init_exp
        m.scale_subp_cuts = scale_subp_cuts
        m.viol_cuts_only = viol_cuts_only
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.tol_zero = tol_zero
        m.prim_cuts_only = prim_cuts_only
        m.prim_cuts_always = prim_cuts_always
        m.prim_cuts_assist = prim_cuts_assist
        m.tol_prim_infeas = tol_prim_infeas
        m.init_sdp_lin = init_sdp_lin
        m.init_sdp_soc = init_sdp_soc
        m.sdp_eig = sdp_eig
        m.sdp_soc = sdp_soc

        m.var_types = Symbol[]
        # m.var_start = Float64[]
        m.num_var_orig = 0
        m.num_con_orig = 0

        m.oa_started = false
        m.best_obj = Inf
        m.mip_obj = -Inf
        m.gap_rel_opt = NaN
        m.status = :NotLoaded

        create_logs!(m)

        return m
    end
end

# Used a lot for scaling PSD cone elements (converting between smat and svec)
const sqrt2 = sqrt(2)
const sqrt2inv = 1/sqrt2

#=========================================================
 MathProgBase functions
=========================================================#

# Verify initial conic data and convert appropriate types and store in Pajarito model
function MathProgBase.loadproblem!(m::PajaritoConicModel, c, A, b, cone_con, cone_var)
    # Verify consistency of conic data
    verify_data(m, c, A, b, cone_con, cone_var)

    # Verify cone compatibility with solver (if solver is not defaultConicsolver: an MPB issue)
    if m.cont_solver != MathProgBase.defaultConicsolver
        # Get cones supported by conic solver
        conic_spec = MathProgBase.supportedcones(m.cont_solver)

        # Pajarito converts rotated SOCs to standard SOCs
        if :SOC in conic_spec
            push!(conic_spec, :SOCRotated)
        end

        # Error if a cone in data is not supported
        for (spec, _) in vcat(cone_con, cone_var)
            if !(spec in conic_spec)
                error("Cones $spec are not supported by the specified conic solver\n")
            end
        end
    end

    # Save original data
    m.num_con_orig = length(b)
    m.num_var_orig = length(c)
    m.c_orig = c
    m.A_orig = A
    m.b_orig = b
    m.cone_con_orig = cone_con
    m.cone_var_orig = cone_var

    m.final_soln = fill(NaN, m.num_var_orig)
    m.status = :Loaded
end

# Store warm-start vector on original variables in Pajarito model
function MathProgBase.setwarmstart!(m::PajaritoConicModel, var_start::Vector{Real})
    error("Warm-starts are not currently implemented in Pajarito (submit an issue)\n")
    # # Check if vector can be loaded
    # if m.status != :Loaded
    #     error("Must specify warm start right after loading problem\n")
    # end
    # if length(var_start) != m.num_var_orig
    #     error("Warm start vector length ($(length(var_start))) does not match number of variables ($(m.num_var_orig))\n")
    # end
    #
    # m.var_start = var_start
end

# Store variable type vector on original variables in Pajarito model
function MathProgBase.setvartype!(m::PajaritoConicModel, var_types::Vector{Symbol})
    if m.status != :Loaded
        error("Must specify variable types right after loading problem\n")
    end
    if length(var_types) != m.num_var_orig
        error("Variable types vector length ($(length(var_types))) does not match number of variables ($(m.num_var_orig))\n")
    end
    if any((var_type -> (var_type != :Bin) && (var_type != :Int) && (var_type != :Cont)), var_types)
        error("Some variable types are not in :Bin, :Int, :Cont\n")
    end
    if !any((var_type -> (var_type == :Bin) || (var_type == :Int)), var_types)
        error("No variables are in :Bin, :Int; use conic solver directly if problem is continuous\n")
    end

    m.var_types = var_types
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    if m.status != :Loaded
        error("Must call optimize! function after loading conic data and setting variable types\n")
    end
    if isempty(m.var_types)
        error("Variable types were not specified; must call setvartype! function\n")
    end

    m.logs[:total] = time()

    # Transform data
    if m.log_level > 1
        @printf "\nTransforming original data..."
    end
    tic()
    (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new) = transform_data(copy(m.c_orig), copy(m.A_orig), copy(m.b_orig), m.cone_con_orig, m.cone_var_orig, m.var_types, m.solve_relax)
    m.logs[:data_trans] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" m.logs[:data_trans]
    end

    # Create conic subproblem data
    if m.log_level > 1
        @printf "\nCreating conic model data..."
    end
    tic()
    (map_rows_sub, cols_cont, cols_int) = create_conicsub_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new)
    m.logs[:data_conic] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" m.logs[:data_conic]
    end

    # Create MIP model
    if m.log_level > 1
        @printf "\nCreating MIP model..."
    end
    tic()
    (v_idxs_soc_relx, r_idx_exp_relx, s_idx_exp_relx, rows_relax_sdp) = create_mip_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, map_rows_sub, cols_cont, cols_int)
    m.logs[:data_mip] += toq()
    if m.log_level > 1
        @printf "...Done %8.2fs\n" m.logs[:data_mip]
    end

    if m.solve_relax
        # Solve relaxed conic problem, proceed with algorithm if optimal or suboptimal, else finish
        if m.log_level > 0
            @printf "\nSolving conic relaxation..."
        end
        tic()
        if m.dualize_relax
            solver_relax = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_relax = m.cont_solver
        end
        model_relax = MathProgBase.ConicModel(solver_relax)
        MathProgBase.loadproblem!(model_relax, c_new, A_new, b_new, cone_con_new, cone_var_new)
        MathProgBase.optimize!(model_relax)
        m.logs[:relax_solve] += toq()
        if m.log_level > 0
            @printf "...Done %8.2fs\n" m.logs[:relax_solve]
        end

        status_relax = MathProgBase.status(model_relax)
        if status_relax == :Infeasible
            warn("Initial conic relaxation status was $status_relax\n")
            m.status = :Infeasible
        elseif status_relax == :Unbounded
            warn("Initial conic relaxation status was $status_relax\n")
            m.status = :UnboundedRelaxation
        elseif (status_relax != :Optimal) && (status_relax != :Suboptimal)
            warn("Apparent conic solver failure with status $status_relax\n")
        else
            obj_relax = MathProgBase.getobjval(model_relax)
            if m.log_level >= 1
                @printf " - Relaxation status    = %14s\n" status_relax
                @printf " - Relaxation objective = %14.6f\n" obj_relax
            end

            # Optionally rescale dual
            dual_conic = MathProgBase.getdual(model_relax)
            if m.scale_subp_cuts
                # Rescale by number of cones / absval of full conic objective
                scale!(dual_conic, (m.num_soc + m.num_exp + m.num_sdp) / (abs(obj_relax) + 1e-5))
            end

            # Add relaxation cuts
            for n in 1:m.num_soc
                add_cut_soc!(m, m.t_soc[n], m.v_soc[n], m.d_soc[n], m.a_soc[n], dual_conic[v_idxs_soc_relx[n]])
            end

            for n in 1:m.num_exp
                add_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], dual_conic[r_idx_exp_relx[n]], dual_conic[s_idx_exp_relx[n]])
            end

            # for n in 1:m.num_sdp
            #     add_cut_sdp!(m, m.dim_sdp[n], m.vars_sdp[n], dual[m.rows_relax_sdp[n]], m.smat[n], false, m.logs[:SDP])
            # end
        end

        # Free the conic model
        if applicable(MathProgBase.freemodel!, model_relax)
            MathProgBase.freemodel!(model_relax)
        end
    end

    if (m.status != :Infeasible) && (m.status != :UnboundedRelaxation)
        tic()
        if m.log_level > 1
            @printf "\nCreating conic subproblem model..."
        end
        if m.dualize_sub
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end
        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        if method_exists(MathProgBase.setbvec!, (typeof(m.model_conic), Vector{Float64}))
            # Can use setbvec! on the conic subproblem model: load it
            m.update_conicsub = true
            MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, m.b_sub, m.cone_con_sub, m.cone_var_sub)
        else
            m.update_conicsub = false
        end
        if m.log_level > 1
            @printf "...Done %8.2fs\n" toq()
        end

        # Initialize and begin iterative or MIP-solver-driven algorithm
        m.logs[:oa_alg] = time()
        m.oa_started = true
        m.new_incumb = false
        m.cache_soln = Set{Vector{Float64}}()

        if m.mip_solver_drives
            if m.log_level > 0
                @printf "\nStarting MIP-solver-driven outer approximation algorithm\n"
            end
            solve_mip_driven!(m)
        else
            if m.log_level > 0
                @printf "\nStarting iterative outer approximation algorithm\n"
            end
            solve_iterative!(m)
        end
        m.logs[:oa_alg] = time() - m.logs[:oa_alg]

        if m.best_obj < Inf
            # Have a best feasible solution, update final solution on original variables
            soln_new = zeros(length(c_new))
            soln_new[cols_int] = m.best_int
            soln_new[cols_cont] = m.best_cont
            m.final_soln = zeros(m.num_var_orig)
            m.final_soln[keep_cols] = soln_new
        end
    end

    # Finish timer and print summary
    m.logs[:total] = time() - m.logs[:total]
    print_finish(m)
end

MathProgBase.numconstr(m::PajaritoConicModel) = m.num_con_orig

MathProgBase.numvar(m::PajaritoConicModel) = m.num_var_orig

MathProgBase.status(m::PajaritoConicModel) = m.status

MathProgBase.getsolvetime(m::PajaritoConicModel) = m.logs[:total]

MathProgBase.getobjval(m::PajaritoConicModel) = m.best_obj

MathProgBase.getobjbound(m::PajaritoConicModel) = m.mip_obj

MathProgBase.getsolution(m::PajaritoConicModel) = m.final_soln


#=========================================================
 Data functions
=========================================================#

# Verify consistency of conic data
function verify_data(m, c, A, b, cone_con, cone_var)
    # Check dimensions of conic problem
    num_con_orig = length(b)
    num_var_orig = length(c)
    if size(A) != (num_con_orig, num_var_orig)
        error("Dimensions of matrix A $(size(A)) do not match lengths of vector b ($(length(b))) and c ($(length(c)))\n")
    end
    if isempty(cone_con) || isempty(cone_var)
        error("Variable or constraint cones are missing\n")
    end

    # Check constraint cones
    inds_con = zeros(Int, num_con_orig)
    for (spec, inds) in cone_con
        if spec == :Free
            error("A cone $spec is in the constraint cones\n")
        end

        if any(inds .> num_con_orig)
            error("Some indices in a constraint cone do not correspond to indices of vector b\n")
        end

        inds_con[inds] += 1
    end
    if any(inds_con .== 0)
        error("Some indices in vector b do not correspond to indices of a constraint cone\n")
    end
    if any(inds_con .> 1)
        error("Some indices in vector b appear in multiple constraint cones\n")
    end

    # Check variable cones
    inds_var = zeros(Int, num_var_orig)
    for (spec, inds) in cone_var
        if any(inds .> num_var_orig)
            error("Some indices in a variable cone do not correspond to indices of vector c\n")
        end

        inds_var[inds] += 1
    end
    if any(inds_var .== 0)
        error("Some indices in vector c do not correspond to indices of a variable cone\n")
    end
    if any(inds_var .> 1)
        error("Some indices in vector c appear in multiple variable cones\n")
    end

    num_soc = 0
    min_soc = 0
    max_soc = 0

    num_rot = 0
    min_rot = 0
    max_rot = 0

    num_exp = 0

    num_sdp = 0
    min_sdp = 0
    max_sdp = 0

    # Verify consistency of cone indices and summarize cone info
    for (spec, inds) in vcat(cone_con, cone_var)
        if isempty(inds)
            error("A cone $spec has no associated indices\n")
        end
        if spec == :SOC
            if length(inds) < 2
                error("A cone $spec has fewer than 2 indices ($(length(inds)))\n")
            end

            num_soc += 1

            if max_soc < length(inds)
                max_soc = length(inds)
            end
            if (min_soc == 0) || (min_soc > length(inds))
                min_soc = length(inds)
            end
        elseif spec == :SOCRotated
            if length(inds) < 3
                error("A cone $spec has fewer than 3 indices ($(length(inds)))\n")
            end

            num_rot += 1

            if max_rot < length(inds)
                max_rot = length(inds)
            end
            if (min_rot == 0) || (min_rot > length(inds))
                min_rot = length(inds)
            end
        elseif spec == :SDP
            if length(inds) < 3
                error("A cone $spec has fewer than 3 indices ($(length(inds)))\n")
            else
                if floor(sqrt(8 * length(inds) + 1)) != sqrt(8 * length(inds) + 1)
                    error("A cone $spec (in SD svec form) does not have a valid (triangular) number of indices ($(length(inds)))\n")
                end
            end

            num_sdp += 1

            if max_sdp < length(inds)
                max_sdp = length(inds)
            end
            if (min_sdp == 0) || (min_sdp > length(inds))
                min_sdp = length(inds)
            end
        elseif spec == :ExpPrimal
            if length(inds) != 3
                error("A cone $spec does not have exactly 3 indices ($(length(inds)))\n")
            end

            num_exp += 1
        end
    end

    m.num_soc = num_soc + num_rot
    m.num_exp = num_exp
    m.num_sdp = num_sdp

    # Print cone info
    if m.log_level <= 1
        return
    end

    @printf "\nCone types summary:"
    @printf "\n%-22s | %-8s | %-8s | %-8s\n" "Cone" "Count" "Min dim" "Max dim"
    if num_soc > 0
        @printf "%22s | %8d | %8d | %8d\n" "Second order" num_soc min_soc max_soc
    end
    if num_rot > 0
        @printf "%22s | %8d | %8d | %8d\n" "Rot. second order" num_rot min_rot max_rot
    end
    if num_exp > 0
        @printf "%22s | %8d | %8d | %8d\n" "Primal exponential" num_exp 3 3
    end
    if num_sdp > 0
        @printf "%22s | %8d | %8d | %8d\n" "Positive semidef." num_sdp min_sdp max_sdp
    end
    flush(STDOUT)
end

# Transform/preprocess data
function transform_data(c_orig, A_orig, b_orig, cone_con_orig, cone_var_orig, var_types, solve_relax)
    A = sparse(A_orig)
    dropzeros!(A)
    (A_I, A_J, A_V) = findnz(A)

    num_con_new = length(b_orig)
    b_new = b_orig
    cone_con_new = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_con_orig]

    num_var_new = 0
    cone_var_new = Tuple{Symbol,Vector{Int}}[]

    old_new_col = zeros(Int, length(c_orig))
    bin_vars_new = Int[]

    vars_nonneg = Int[]
    vars_nonpos = Int[]
    vars_free = Int[]
    for (spec, cols) in cone_var_orig
        # Ignore zero variable cones
        if spec != :Zero
            vars_nonneg = Int[]
            vars_nonpos = Int[]
            vars_free = Int[]

            for j in cols
                if var_types[j] == :Bin
                    # Put binary vars in NonNeg var cone, unless the original var cone was NonPos in which case the binary vars are fixed at zero
                    if spec != :NonPos
                        num_var_new += 1
                        old_new_col[j] = num_var_new
                        push!(vars_nonneg, j)
                        push!(bin_vars_new, j)
                    end
                else
                    # Put non-binary vars in NonNeg or NonPos or Free var cone
                    num_var_new += 1
                    old_new_col[j] = num_var_new
                    if spec == :NonNeg
                        push!(vars_nonneg, j)
                    elseif spec == :NonPos
                        push!(vars_nonpos, j)
                    else
                        push!(vars_free, j)
                    end
                end
            end

            if !isempty(vars_nonneg)
                push!(cone_var_new, (:NonNeg, old_new_col[vars_nonneg]))
            end
            if !isempty(vars_nonpos)
                push!(cone_var_new, (:NonPos, old_new_col[vars_nonpos]))
            end
            if !isempty(vars_free)
                push!(cone_var_new, (:Free, old_new_col[vars_free]))
            end

            if (spec != :Free) && (spec != :NonNeg) && (spec != :NonPos)
                # Convert nonlinear var cone to constraint cone
                push!(cone_con_new, (spec, collect((num_con_new + 1):(num_con_new + length(cols)))))
                for j in cols
                    num_con_new += 1
                    push!(A_I, num_con_new)
                    push!(A_J, j)
                    push!(A_V, -1.)
                    push!(b_new, 0.)
                end
            end
        end
    end

    A = sparse(A_I, A_J, A_V, num_con_new, length(c_orig))
    keep_cols = find(old_new_col)
    c_new = c_orig[keep_cols]
    A = A[:, keep_cols]
    var_types_new = var_types[keep_cols]

    # Convert SOCRotated cones to SOC cones (MathProgBase definitions)
    # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt2*x) in SOC, y >= 0, z >= 0
    socr_rows = Vector{Int}[]
    for n_cone in 1:length(cone_con_new)
        (spec, rows) = cone_con_new[n_cone]
        if spec == :SOCRotated
            cone_con_new[n_cone] = (:SOC, rows)
            push!(socr_rows, rows)
        end
    end

    (A_I, A_J, A_V) = findnz(A)
    row_to_nzind = map(_ -> Int[], 1:num_con_new)
    for (ind, i) in enumerate(A_I)
        push!(row_to_nzind[i], ind)
    end

    for rows in socr_rows
        inds_1 = row_to_nzind[rows[1]]
        inds_2 = row_to_nzind[rows[2]]

        # Add new constraint cones for y >= 0, z >= 0
        push!(cone_con_new, (:NonNeg, collect((num_con_new + 1):(num_con_new + 2))))

        append!(A_I, fill((num_con_new + 1), length(inds_1)))
        append!(A_J, A_J[inds_1])
        append!(A_V, A_V[inds_1])
        push!(b_new, b_new[rows[1]])

        append!(A_I, fill((num_con_new + 2), length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])
        push!(b_new, b_new[rows[2]])

        num_con_new += 2

        # Use old constraint cone SOCRotated for (y+z,-y+z,sqrt2*x) in SOC
        append!(A_I, fill(rows[1], length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])
        b_new[rows[1]] += b_new[rows[2]]

        append!(A_I, fill(rows[2], length(inds_1)))
        append!(A_J, A_J[inds_1])
        append!(A_V, -A_V[inds_1])
        b_new[rows[2]] -= b_new[rows[1]]

        for i in rows[3:end]
            for ind in row_to_nzind[i]
                A_V[ind] *= sqrt2
            end
        end
        b_new[rows[2:end]] .*= sqrt2
    end

    if solve_relax
        # Preprocess to tighten bounds on binary and integer variables in conic relaxation
        # Detect isolated row nonzeros with nonzero b
        row_slck_count = zeros(Int, num_con_new)
        for (ind, i) in enumerate(A_I)
            if (A_V[ind] != 0.) && (b_new[i] != 0.)
                if row_slck_count[i] == 0
                    row_slck_count[i] = ind
                elseif row_slck_count[i] > 0
                    row_slck_count[i] = -1
                end
            end
        end

        bin_set_upper = falses(length(bin_vars_new))
        j = 0
        type_j = :Cont
        bound_j = 0.0

        # For each bound-type constraint, tighten by rounding
        for (spec, rows) in cone_con_new
            if (spec != :NonNeg) && (spec != :NonPos)
                continue
            end

            for i in rows
                if row_slck_count[i] > 0
                    # Isolated variable x_j with b_i - a_ij*x_j in spec, b_i & a_ij nonzero
                    j = A_J[row_slck_count[i]]
                    type_j = var_types[keep_cols[j]]
                    bound_j = b_new[i] / A_V[row_slck_count[i]]

                    if (spec == :NonNeg) && (A_V[row_slck_count[i]] > 0) || (spec == :NonPos) && (A_V[row_slck_count[i]] < 0)
                        # Upper bound: b_i/a_ij >= x_j
                        if (type_j == :Bin) && (bound_j >= 1.)
                            # Tighten binary upper bound to 1
                            if spec == :NonNeg
                                # 1 >= x_j
                                b_new[i] = 1.
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -1 <= -x_j
                                b_new[i] = -1.
                                A_V[row_slck_count[i]] = -1.
                            end

                            bin_set_upper[j] = true
                        elseif type_j != :Cont
                            # Tighten binary or integer upper bound by rounding down
                            # TODO this may cause either fixing or infeasibility: detect this and remove variable (at least for binary)
                            if spec == :NonNeg
                                # floor >= x_j
                                b_new[i] = floor(bound_j)
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -floor <= -x_j
                                b_new[i] = -floor(bound_j)
                                A_V[row_slck_count[i]] = -1.
                            end

                            if type_j == :Bin
                                bin_set_upper[j] = true
                            end
                        end
                    else
                        # Lower bound: b_i/a_ij <= x_j
                        if type_j != :Cont
                            # Tighten binary or integer lower bound by rounding up
                            # TODO this may cause either fixing or infeasibility: detect this and remove variable (at least for binary)
                            if spec == :NonPos
                                # ceil <= x_j
                                b_new[i] = ceil(bound_j)
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -ceil >= -x_j
                                b_new[i] = -ceil(bound_j)
                                A_V[row_slck_count[i]] = -1.
                            end
                        end
                    end
                end
            end
        end

        # For any binary variables without upper bound set, add 1 >= x_j to constraint cones
        num_con_prev = num_con_new
        for ind in 1:length(bin_vars_new)
            if !bin_set_upper[ind]
                num_con_new += 1
                push!(A_I, num_con_new)
                push!(A_J, bin_vars_new[ind])
                push!(A_V, 1.)
                push!(b_new, 1.)
            end
        end
        if num_con_new > num_con_prev
            push!(cone_con_new, (:NonNeg, collect((num_con_prev + 1):num_con_new)))
        end
    end

    A_new = sparse(A_I, A_J, A_V, num_con_new, num_var_new)
    dropzeros!(A_new)

    return (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new)
end

# Create conic subproblem data
function create_conicsub_data!(m, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int64}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol})
    # Build new subproblem variable cones by removing integer variables
    cols_cont = Int[]
    cols_int = Int[]
    num_cont = 0
    cone_var_sub = Tuple{Symbol,Vector{Int}}[]

    for (spec, cols) in cone_var_new
        cols_cont_new = Int[]
        for j in cols
            if var_types_new[j] == :Cont
                push!(cols_cont, j)
                num_cont += 1
                push!(cols_cont_new, num_cont)
            else
                push!(cols_int, j)
            end
        end
        if !isempty(cols_cont_new)
            push!(cone_var_sub, (spec, cols_cont_new))
        end
    end

    # Determine "empty" rows with no nonzero coefficients on continuous variables
    (A_cont_I, _, A_cont_V) = findnz(A_new[:, cols_cont])
    num_con_new = size(A_new, 1)
    rows_nz = falses(num_con_new)
    for (i, v) in zip(A_cont_I, A_cont_V)
        if !rows_nz[i] && (v != 0)
            rows_nz[i] = true
        end
    end

    # Build new subproblem constraint cones by removing empty rows
    num_full = 0
    rows_full = Int[]
    cone_con_sub = Tuple{Symbol,Vector{Int}}[]
    map_rows_sub = Vector{Int}(num_con_new)

    for (spec, rows) in cone_con_new
        if (spec == :Zero) || (spec == :NonNeg) || (spec == :NonPos)
            rows_full_new = Int[]
            for i in rows
                if rows_nz[i]
                    push!(rows_full, i)
                    num_full += 1
                    push!(rows_full_new, num_full)
                end
            end
            if !isempty(rows_full_new)
                push!(cone_con_sub, (spec, rows_full_new))
            end
        else
            map_rows_sub[rows] = collect((num_full + 1):(num_full + length(rows)))
            push!(cone_con_sub, (spec, collect((num_full + 1):(num_full + length(rows)))))
            append!(rows_full, rows)
            num_full += length(rows)
        end
    end

    # Store conic data
    m.cone_var_sub = cone_var_sub
    m.cone_con_sub = cone_con_sub

    # Build new subproblem A, b, c data by removing empty rows and integer variables
    m.A_sub_cont = A_new[rows_full, cols_cont]
    m.A_sub_int = A_new[rows_full, cols_int]
    m.b_sub = b_new[rows_full]
    m.c_sub_cont = c_new[cols_cont]
    m.c_sub_int = c_new[cols_int]

    return (map_rows_sub, cols_cont, cols_int)
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int64}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol}, map_rows_sub::Vector{Int}, cols_cont::Vector{Int}, cols_int::Vector{Int})
    # Initialize JuMP model for MIP outer approximation problem
    model_mip = JuMP.Model(solver=m.mip_solver)

    # Create variables and set types
    x_all = @variable(model_mip, [1:length(var_types_new)])
    for j in cols_int
        setcategory(x_all[j], var_types_new[j])
    end

    # Set objective function
    @objective(model_mip, :Min, dot(c_new, x_all))

    # Add variable cones to MIP
    for (spec, cols) in cone_var_new
        if spec == :NonNeg
            for j in cols
                setname(x_all[j], "v$(j)")
                setlowerbound(x_all[j], 0.)
            end
        elseif spec == :NonPos
            for j in cols
                setname(x_all[j], "v$(j)")
                setupperbound(x_all[j], 0.)
            end
        elseif spec == :Free
            for j in cols
                setname(x_all[j], "v$(j)")
            end
        elseif spec == :Zero
            error("Bug: Zero cones should have been removed by transform data function (submit an issue)\n")
        end
    end

    # Allocate data for nonlinear cones
    # SOC data
    v_idxs_soc_relx = Vector{Vector{Int}}(m.num_soc)
    t_idx_soc_subp = Vector{Int}(m.num_soc)
    v_idxs_soc_subp = Vector{Vector{Int}}(m.num_soc)
    t_soc = Vector{JuMP.AffExpr}(m.num_soc)
    v_soc = Vector{Vector{JuMP.AffExpr}}(m.num_soc)
    d_soc = Vector{Vector{JuMP.Variable}}(m.num_soc)
    a_soc = Vector{Vector{JuMP.Variable}}(m.num_soc)

    # Exp data
    r_idx_exp_relx = Vector{Int}(m.num_exp)
    s_idx_exp_relx = Vector{Int}(m.num_exp)
    r_idx_exp_subp = Vector{Int}(m.num_exp)
    s_idx_exp_subp = Vector{Int}(m.num_exp)
    t_idx_exp_subp = Vector{Int}(m.num_exp)
    r_exp = Vector{JuMP.AffExpr}(m.num_exp)
    s_exp = Vector{JuMP.AffExpr}(m.num_exp)
    t_exp = Vector{JuMP.AffExpr}(m.num_exp)

    # PSD data
    rows_relax_sdp = Vector{Vector{Int}}(m.num_sdp)
    # rows_sub_sdp = Vector{Vector{Int}}(num_sdp)
    # dim_sdp = Vector{Int}(num_sdp)
    # # vars_svec_sdp = Vector{Vector{JuMP.Variable}}(num_sdp)
    # vars_sdp = Vector{Array{JuMP.AffExpr,2}}(num_sdp)
    # smat = Vector{Array{Float64,2}}(num_sdp)

    # Set up a SOC cone in the MIP
    function add_soc!(t, v, d, a, dim)
        # Set bounds
        @constraint(model_mip, t >= 0)

        if m.soc_disagg
            # Add disaggregated SOC constraint
            # t >= sum(2*d_j)
            # Scale by 2
            @constraint(model_mip, 2*(t - 2*sum(d)) >= 0)
        end

        if m.soc_abslift
            # Add absolute value SOC constraints
            # a_j >= v_j, a_j >= -v_j
            # Scale by 2
            for j in 1:dim
                @constraint(model_mip, 2*(a[j] - v[j]) >= 0)
                @constraint(model_mip, 2*(a[j] + v[j]) >= 0)
            end
        end

        if m.init_soc_one
            # Add initial L_1 SOC linearizations if using disaggregation or absvalue lifting (otherwise no polynomial number of cuts)
            # t >= 1/sqrt(dim)*sum(|v_j|)
            if m.soc_disagg && m.soc_abslift
                # Using disaggregation and absvalue lifting
                # 1/dim*t + 2*d_j >= 2/sqrt(dim)*a_j, all j
                # Scale by 2*dim
                for j in 1:dim
                    @constraint(model_mip, 2*(t + 2*dim*d[j] - 2*sqrt(dim)*a[j]) >= 0)
                end
            elseif m.soc_disagg
                # Using disaggregation only
                # 1/dim*t + 2*d_j >= 2/sqrt(dim)*|v_j|, all j
                # Scale by 2*dim
                for j in 1:dim
                    @constraint(model_mip, 2*(t + 2*dim*d[j] - 2*sqrt(dim)*v[j]) >= 0)
                    @constraint(model_mip, 2*(t + 2*dim*d[j] + 2*sqrt(dim)*v[j]) >= 0)
                end
            else
                # Using absvalue lifting only
                # t >= 1/sqrt(dim)*sum(a_j)
                # Scale by 2
                @constraint(model_mip, 2*(t - 1/sqrt(dim)*sum(a)) >= 0)
            end
        end

        if m.init_soc_inf
            # Add initial L_inf SOC linearizations
            # t >= |v_j|, all j
            if m.soc_disagg && m.soc_abslift
                # Using disaggregation and absvalue lifting
                # t + d_j >= 2*a_j, all j
                # Scale by 2*dim
                for j in 1:dim
                    @constraint(model_mip, 2*dim*(t + 2*d[j] - 2*a[j]) >= 0)
                end
            elseif m.soc_disagg
                # Using disaggregation only
                # t + d_j >= 2*|v_j|, all j
                # Scale by 2*dim
                for j in 1:dim
                    @constraint(model_mip, 2*dim*(t + 2*d[j] - 2*v[j]) >= 0)
                    @constraint(model_mip, 2*dim*(t + 2*d[j] + 2*v[j]) >= 0)
                end
            elseif m.soc_abslift
                # Using absvalue lifting only
                # t >= a_j, all j
                # Scale by 2
                for j in 1:dim
                    @constraint(model_mip, 2*(t - a[j]) >= 0)
                end
            else
                # Using no lifting
                # t >= |v_j|, all j
                # Scale by 2
                for j in 1:dim
                    @constraint(model_mip, 2*(t - v[j]) >= 0)
                    @constraint(model_mip, 2*(t + v[j]) >= 0)
                end
            end
        end
    end

    # Set up a ExpPrimal cone in the MIP
    function add_exp!(r, s, t)
        # Set bounds
        @constraint(model_mip, s >= 0)
        @constraint(model_mip, t >= 0)

        if m.init_exp
            # Add initial exp cuts using dual exp cone linearizations
            # (u,v,w) in ExpDual <-> exp(1)*w >= -u*exp(v/u), w >= 0, u < 0
            # at u = -1; v = -1, -1/2, -1/5, 0, 1/5, 1/2, 1; z = exp(-v-1)
            for v in [-1., -0.5, -0.2, 0., 0.2, 0.5, 1.]
                @constraint(model_mip, -r + v*s + exp(-v-1)*w >= 0)
            end
        end
    end

    # Set up a SDP cone in the MIP
    # function add_sdp!(n_sdp, rows, vars)
    #
    #     # dim = round(Int, sqrt(1/4 + 2 * length(rows)) - 1/2) # smat space dimension
    #
    #
    #
    #     # dim_sdp[n_sdp] = dim
    #     # rows_relax_sdp[n_sdp] = rows
    #     # rows_sub_sdp[n_sdp] = map_rows_sub[rows]
    #     # # vars_svec_sdp[n_sdp] = vars
    #     # smat[n_sdp] = zeros(dim, dim)
    #     # vars_smat = Array{JuMP.AffExpr,2}(dim, dim)
    #     # vars_sdp[n_sdp] = vars_smat
    #     #
    #     # # Set up smat arrays and set bounds
    #     # kSD = 1
    #     # for jSD in 1:dim, iSD in jSD:dim
    #     #     if jSD == iSD
    #     #         @constraint(model_mip, vars[kSD] >= 0)
    #     #         vars_smat[iSD, jSD] = vars[kSD]
    #     #     else
    #     #         vars_smat[iSD, jSD] = vars_smat[jSD, iSD] = sqrt2inv * vars[kSD]
    #     #     end
    #     #     kSD += 1
    #     # end
    #     #
    #     # # what about a lifting for abs value
    #     #
    #     # # Add initial (linear or SOC) SDP outer approximation cuts
    #     # for jSD in 1:dim, iSD in (jSD + 1):dim
    #     #     if m.init_sdp_soc
    #     #         # Add initial rotated SOC for off-diagonal element to enforce 2x2 principal submatrix PSDness
    #     #         # Use norm and transformation from RSOC to SOC
    #     #         # yz >= ||x||^2, y,z >= 0 <==> norm2(2x, y-z) <= y + z
    #     #         @constraint(model_mip, vars_smat[iSD, iSD] + vars_smat[jSD, jSD] >= norm(JuMP.AffExpr[(2. * vars_smat[iSD, jSD]), (vars_smat[iSD, iSD] - vars_smat[jSD, jSD])]))
    #     #     elseif m.init_sdp_lin
    #     #         # Add initial SDP linear cuts based on linearization of 3-dim rotated SOCs that enforce 2x2 principal submatrix PSDness (essentially the dual of SDSOS)
    #     #         # 2|m_ij| <= m_ii + m_jj, where m_kk is scaled by sqrt2 in smat space
    #     #         @constraint(model_mip, vars_smat[iSD, iSD] + vars_smat[jSD, jSD] >= 2. * vars_smat[iSD, jSD])
    #     #         @constraint(model_mip, vars_smat[iSD, iSD] + vars_smat[jSD, jSD] >= -2. * vars_smat[iSD, jSD])
    #     #     end
    #     # end
    # end

    n_soc = 0
    n_exp = 0
    n_sdp = 0
    @expression(model_mip, lhs_expr, b_new - A_new * x_all)

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    for (spec, rows) in cone_con_new
        if spec == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0)
        elseif spec == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)
        elseif spec == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)
        elseif spec == :SOC
            if m.soc_in_mip
                # If putting SOCs in the MIP directly, don't need to use other SOC infrastructure
                @constraint(model_mip, t >= norm(v))
                continue
            end

            # Set up a SOC
            # (t,v) in SOC <-> t >= norm(v)
            n_soc += 1
            v_idxs = rows[2:end]
            dim = length(v_idxs)
            v_idxs_soc_relx[n_soc] = v_idxs
            t_idx_soc_subp[n_soc] = map_rows_sub[rows[1]]
            v_idxs_soc_subp[n_soc] = map_rows_sub[v_idxs]

            if m.soc_disagg
                # Add disaggregated SOC variables d_j
                # 2*d_j >= v_j^2/t, all j
                d = @variable(model_mip, [j in 1:dim], lowerbound=0)
                for j in 1:dim
                    setname(d[j], "d$(j)_soc$(n_soc)")
                end
            else
                d = Vector{JuMP.Variable}()
            end

            if m.soc_abslift
                # Add absolute value SOC variables a_j
                # a_j >= |v_j|
                a = @variable(model_mip, [j in 1:dim], lowerbound=0)
                for j in 1:dim
                    setname(a[j], "a$(j)_soc$(n_soc)")
                end
            else
                a = Vector{JuMP.Variable}()
            end

            t_soc[n_soc] = t = lhs_expr[rows[1]]
            v_soc[n_soc] = v = lhs_expr[v_idxs]
            d_soc[n_soc] = d
            a_soc[n_soc] = a

            add_soc!(t, v, d, a, dim)
        elseif spec == :ExpPrimal
            # Set up a ExpPrimal cone
            # (r,s,t) in ExpPrimal <-> t >= s*exp(r/s)
            n_exp += 1
            r_idx_exp_relx[n_soc] = rows[1]
            s_idx_exp_relx[n_soc] = rows[2]
            r_idx_exp_subp[n_soc] = map_rows_sub[rows[1]]
            s_idx_exp_subp[n_soc] = map_rows_sub[rows[2]]
            t_idx_exp_subp[n_soc] = map_rows_sub[rows[3]]

            r_exp[n_soc] = r = lhs_expr[rows[1]]
            s_exp[n_soc] = s = lhs_expr[rows[2]]
            t_exp[n_soc] = t = lhs_expr[rows[3]]

            add_exp!(r, s, t)
        elseif spec == :SDP
            n_sdp += 1
            add_sdp!(n_sdp, rows, lhs_expr[rows])
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_int = x_all[cols_int]
    m.x_cont = x_all[cols_cont]
    # @show model_mip

    m.v_idxs_soc_subp = v_idxs_soc_subp
    m.t_idx_soc_subp = t_idx_soc_subp
    m.t_soc = t_soc
    m.v_soc = v_soc
    m.d_soc = d_soc
    m.a_soc = a_soc

    m.r_idx_exp_subp = r_idx_exp_subp
    m.s_idx_exp_subp = s_idx_exp_subp
    m.t_idx_exp_subp = t_idx_exp_subp
    m.r_exp = r_exp
    m.s_exp = s_exp
    m.t_exp = t_exp

    # m.rows_sub_sdp = rows_sub_sdp
    # m.dim_sdp = dim_sdp
    # # m.vars_svec_sdp = vars_svec_sdp
    # m.vars_sdp = vars_sdp
    # m.smat = smat

    return (v_idxs_soc_relx, r_idx_exp_relx, s_idx_exp_relx, rows_relax_sdp)
end


#=========================================================
 Algorithms
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m)
    count_subopt = 0

    while true
        reset_cone_logs!(m)

        if count_subopt < m.mip_subopt_count
            # Solve is a partial solve: use subopt MIP solver, trust that user has provided reasonably small time limit
            setsolver(m.model_mip, m.mip_subopt_solver)
            count_subopt += 1
        else
            # Solve is a full solve: use full MIP solver with remaining time limit
            if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
                MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(0., m.timeout - (time() - m.logs[:total])))
            end
            setsolver(m.model_mip, m.mip_solver)
            count_subopt = 0
        end

        # Solve MIP
        tic()
        status_mip = solve(m.model_mip)#, suppress_warnings=true)
        m.logs[:mip_solve] += toq()
        m.logs[:n_mip] += 1

        if (status_mip == :Infeasible) || (status_mip == :InfeasibleOrUnbounded)
            # Stop if infeasible
            m.status = :Infeasible
            break
        elseif status_mip == :Unbounded
            # Stop if unbounded (initial conic relax solve should detect this)
            if m.solve_relax
                warn("MIP solver returned status $status_mip, which suggests that the initial subproblem cuts added were too weak\n")
            else
                warn("MIP solver returned status $status_mip, because the initial conic relaxation was not solved\n")
            end
            m.status = :CutsFailure
            break
        elseif (status_mip == :UserLimit) || (status_mip == :Optimal)
            # Update OA bound if MIP bound is better than current OA bound
            mip_obj_bound = MathProgBase.getobjbound(m.model_mip)
            if mip_obj_bound > m.mip_obj
                m.mip_obj = mip_obj_bound

                # Calculate relative outer approximation gap, finish if satisfy optimality gap condition
                m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
                if m.gap_rel_opt < m.rel_gap
                    print_gap(m)
                    m.status = :Optimal
                    break
                end
            end

            # Timeout if MIP reached time limit
            if status_mip == :UserLimit && ((time() - m.logs[:total]) > (m.timeout - 0.01))
                m.status = :UserLimit
                break
            end

            # If solver doesn't have a feasible solution, must immediately try solving to optimality
            if isnan(getobjectivevalue(m.model_mip))
                count_subopt = m.mip_subopt_count
                warn("Solution has NaN values, proceeding to next optimal MIP solve\n")
            end
        else
            warn("MIP solver returned status $status_mip, which Pajarito does not handle (please submit an issue)\n")
            m.status = :MIPFailure
            break
        end

        # Solve new conic subproblem, update incumbent solution if feasible
        (is_repeat, is_viol_subp) = add_subp_incumb_cuts!(m)

        if m.new_incumb
            # Have a new incumbent from conic solver, calculate relative outer approximation gap, finish if satisfy optimality gap condition
            m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
            if m.gap_rel_opt < m.rel_gap
                print_gap(m)
                m.status = :Optimal
                break
            end
        end

        if !is_viol_subp || m.prim_cuts_always
            # No violated subproblem cuts added, or always adding primal cuts
            # Check feasibility and add primal cuts if primal cuts for convergaence assistance
            (is_infeas, is_viol_prim) = add_prim_feas_cuts!(m, m.prim_cuts_assist)

            if !is_infeas
                # MIP solver solution is conic-feasible, check if it is a new incumbent
                soln_int = getvalue(m.x_int)
                soln_cont = getvalue(m.x_cont)
                obj_full = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_cont)

                if obj_full < m.best_obj
                    # Save new incumbent info
                    m.best_obj = obj_full
                    m.best_int = soln_int
                    m.best_cont = soln_cont

                    # Calculate relative outer approximation gap, finish if satisfy optimality gap condition
                    m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
                    if m.gap_rel_opt < m.rel_gap
                        print_gap(m)
                        m.status = :Optimal
                        break
                    end
                end
            elseif is_repeat && !is_viol_prim
                # Integer solution has repeated, conic solution is infeasible, and no violated primal cuts were added
                if count_subopt == 0
                    # Solve was optimal solve, so nothing more we can do
                    if m.prim_cuts_assist
                        warn("No violated subproblem cuts or primal cuts were added on conic-infeasible OA solution (this should not happen: please submit an issue)\n")
                    else
                        warn("No violated subproblem cuts or primal cuts were added on conic-infeasible OA solution (try using prim_cuts_assist = true)\n")
                    end
                    m.status = :CutsFailure
                    break
                end

                # Try solving next MIP to optimality, if that doesn't help then we will fail next iteration
                warn("Integer solution has repeated, solving next MIP to optimality\n")
                count_subopt = m.mip_subopt_count
            end
        end

        print_gap(m)

        # Finish if exceeded timeout option
        if (time() - m.logs[:oa_alg]) > m.timeout
            m.status = :UserLimit
            break
        end

        # Give the best feasible solution to the MIP as a warm-start
        if m.pass_mip_sols && m.new_incumb
            set_best_soln!(m)
            m.new_incumb = false
        end
    end
end

# Solve the MIP model using MIP-solver-driven callback algorithm
function solve_mip_driven!(m)
    if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
        MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(0., m.timeout - (time() - m.logs[:total])))
        setsolver(m.model_mip, m.mip_solver)
    end

    # Add lazy cuts callback to add dual and primal conic cuts
    function callback_lazy(cb)
        m.cb_lazy = cb
        reset_cone_logs!(m)

        # Solve new conic subproblem, update incumbent solution if feasible
        (is_repeat, is_viol_subp) = add_subp_incumb_cuts!(m)

        # Finish if any violated subproblem cuts were added and not using primal cuts always
        if is_viol_subp && !m.prim_cuts_always
            return
        end

        # Check feasibility of current solution, try to add violated primal cuts if using primal cuts for convergence assistance
        (is_infeas, is_viol_prim) = add_prim_feas_cuts!(m, m.prim_cuts_assist)

        # Finish if any violated cuts have been added or if solution is conic feasible
        if !is_infeas || is_viol_subp || is_viol_prim
            return
        end

        # No violated cuts could be added on conic infeasible solution: fail
        # (Don't need to fail if solution doesn't improve MIP's best solution value, but this is probably rare or impossible depending on MIP solver behavior)
        if m.prim_cuts_assist
            warn("No violated subproblem cuts or primal cuts were added on conic-infeasible OA solution (this should not happen: please submit an issue)\n")
        else
            warn("No violated subproblem cuts or primal cuts were added on conic-infeasible OA solution (try using prim_cuts_assist = true)\n")
        end
        m.status = :CutsFailure
        return JuMP.StopTheSolver
    end
    addlazycallback(m.model_mip, callback_lazy)

    if m.pass_mip_sols
        # Add heuristic callback to give MIP solver feasible solutions from conic solves
        function callback_heur(cb)
            # If have a new best feasible solution since last heuristic solution added, set MIP solution to the new best feasible solution
            if m.new_incumb
                m.cb_heur = cb
                set_best_soln!(m)
                addsolution(cb)
                m.new_incumb = false
            end
        end
        addheuristiccallback(m.model_mip, callback_heur)
    end

    # Start MIP solver
    m.logs[:mip_solve] = time()
    status_mip = solve(m.model_mip)#, suppress_warnings=true)
    m.logs[:mip_solve] = time() - m.logs[:mip_solve]

    if (status_mip == :Infeasible) || (status_mip == :InfeasibleOrUnbounded)
        m.status = :Infeasible
        return
    elseif status_mip == :Unbounded
        if m.solve_relax
            warn("MIP solver returned status $status_mip, which suggests that the initial subproblem cuts added were too weak\n")
        else
            warn("MIP solver returned status $status_mip, because the initial conic relaxation was not solved\n")
        end
        m.status = :CutsFailure
        return
    elseif status_mip == :UserLimit
        # Either a timeout, or a cuts failure terminated the MIP solver
        m.mip_obj = getobjbound(m.model_mip)
        if isfinite(m.best_obj)
            # We have a feasible solution
            m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
        end
        if m.status != :CutsFailure
            m.status = status_mip
        end
        return
    elseif status_mip == :Optimal
        # Check if conic solver solution (if exists) satisfies gap condition, if so, use that solution, else use MIP solver's solution
        # (Since we didn't stop the MIP solver due to cuts failure, the MIP solution should be conic feasible)
        m.mip_obj = getobjbound(m.model_mip)
        if isfinite(m.best_obj)
            # We have a feasible solution from conic solver
            m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
            if m.gap_rel_opt < m.rel_gap
                # Solution satisfies gap
                m.status = :Optimal
                return
            end
        end

        # Use MIP solver's solution
        m.best_int = getvalue(m.x_int)
        m.best_cont = getvalue(m.x_cont)
        m.best_obj = dot(m.c_sub_int, m.best_int) + dot(m.c_sub_cont, m.best_cont)
        m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
        if m.gap_rel_opt < m.rel_gap
            m.status = :Optimal
        else
            m.status = :Suboptimal
        end
        return
    else
        warn("MIP solver returned status $status_mip, which Pajarito does not handle (please submit an issue)\n")
        m.status = :MIPFailure
        return
    end
end


#=========================================================
 Warm-starting / heuristic functions
=========================================================#

# Construct and warm-start MIP solution using best solution
function set_best_soln!(m)
    set_soln!(m, m.x_int, m.best_int)
    set_soln!(m, m.x_cont, m.best_cont)

    for n in 1:m.num_soc
        if m.soc_disagg
            set_d_soln!(m, m.d_soc[n], m.best_slck[m.v_idxs_soc_subp[n]], m.best_slck[m.t_idx_soc_subp[n]])
        end
        if m.soc_abslift
            set_a_soln!(m, m.a_soc[n], m.best_slck[m.v_idxs_soc_subp[n]])
        end
    end

    #TODO other cones
end

# Call setvalue or setsolutionvalue solution for a vector of variables and a solution vector
function set_soln!(m, vars::Vector{JuMP.Variable}, soln::Vector{Float64})
    if m.mip_solver_drives && m.oa_started
        for j in 1:length(vars)
            setsolutionvalue(m.cb_heur, vars[j], soln[j])
        end
    else
        for j in 1:length(vars)
            setvalue(vars[j], soln[j])
        end
    end
end

# Call setvalue or setsolutionvalue solution for a vector of SOC disaggregated variables
function set_d_soln!(m, d::Vector{JuMP.Variable}, v_slck::Vector{Float64}, t_slck::Float64)
    if m.mip_solver_drives && m.oa_started
        for j in 1:length(d)
            setsolutionvalue(m.cb_heur, d[j], (v_slck[j]^2/(2.*t_slck)))
        end
    else
        for j in 1:length(d)
            setvalue(d[j], (v_slck[j]^2/(2.*t_slck)))
        end
    end
end

# Call setvalue or setsolutionvalue solution for a vector of SOC absvalue lifting variables
function set_a_soln!(m, a::Vector{JuMP.Variable}, v_slck::Vector{Float64})
    if m.mip_solver_drives && m.oa_started
        for j in 1:length(a)
            setsolutionvalue(m.cb_heur, a[j], abs(v_slck[j]))
        end
    else
        for j in 1:length(a)
            setvalue(a[j], abs(v_slck[j]))
        end
    end
end

# Transform svec vector into symmetric smat matrix
function make_smat!(svec::Vector{Float64}, smat::Array{Float64,2})
    dim = size(smat, 1)
    kSD = 1
    for jSD in 1:dim, iSD in jSD:dim
        if jSD == iSD
            smat[iSD, jSD] = svec[kSD]
        else
            smat[iSD, jSD] = smat[jSD, iSD] = sqrt2inv * svec[kSD]
        end
        kSD += 1
    end
    return smat
end


#=========================================================
 K^* cuts functions
=========================================================#

# Solve the subproblem for the current integer solution, add new incumbent conic solution if feasible and best, Add K* cuts from subproblem dual solution
function add_subp_incumb_cuts!(m)
    # Get current integer solution and check if it is new
    soln_int = getvalue(m.x_int)
    if m.round_mip_sols
        # Round the integer values
        soln_int = map!(round, soln_int)
    end
    if soln_int in m.cache_soln
        # Integer solution has been seen before, cannot get new subproblem cuts
        m.logs[:n_repeat] += 1
        return (true, false)
    end

    # Integer solution is new
    push!(m.cache_soln, soln_int)

    # Calculate new b vector from integer solution and solve conic subproblem model
    b_sub_int = m.b_sub - m.A_sub_int*soln_int
    (status_conic, soln_conic, dual_conic) = solve_subp!(m, b_sub_int)

    # Determine cut scaling factors and check if have new feasible incumbent solution
    if status_conic == :Infeasible
        # Subproblem infeasible
        if m.scale_subp_cuts
            # First check infeasible ray has negative value
            ray_value = vecdot(dual_conic, b_sub_int)
            if ray_value > -m.tol_zero
                warn("Serious conic solver failure: returned status $status_conic but b'y is not sufficiently negative for infeasible ray y (this should not happen: please submit an issue)\n")
                return (false, false)
            end

            # Rescale by number of cones / value of ray
            scale!(dual_conic, (m.num_soc + m.num_exp + m.num_sdp) / ray_value)
        end
    elseif (status_conic == :Optimal) || (status_conic == :Suboptimal)
        # Subproblem feasible
        # Note: suboptimal is a poorly defined status for conic solvers, this status should be rare (whether or not the dual is valid, the K* cuts are always valid)
        # Clean zeros and calculate full objective value
        clean_zeros!(m, soln_conic)
        obj_full = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_conic)

        if m.scale_subp_cuts
            # Rescale by number of cones / abs(objective + 1e-5)
            scale!(dual_conic, (m.num_soc + m.num_exp + m.num_sdp) / (abs(obj_full) + 1e-5))
        end

        if obj_full < m.best_obj
            # Conic solver solution is a new incumbent
            # Note: perhaps should check feasibility of conic solution with respect to our primal inf tol, but it should be satisfied except rarely
            # warn("Conic solver solution does not satisfy Pajarito's primal infeasibility tolerances (try increasing tol_prim_infeas or decreasing feasibility tolerances on the conic solver)\n")
            # m.logs[:n_feas] += 1
            m.best_obj = obj_full
            m.best_int = soln_int
            m.best_cont = soln_conic
            m.best_slck = b_sub_int - m.A_sub_cont * soln_conic
            m.new_incumb = true
        end
    else
        # Status not handled, cannot add subproblem cuts
        warn("Conic solver failure: returned status $status_conic\n")
        return (false, false)
    end

    # If not using subproblem cuts, return
    if m.prim_cuts_only
        return (false, false)
    end

    # Add subproblem cuts for each cone
    is_viol_subp = false

    for n in 1:m.num_soc
        if add_cut_soc!(m, m.t_soc[n], m.v_soc[n], m.d_soc[n], m.a_soc[n], dual_conic[m.v_idxs_soc_subp[n]])
            is_viol_subp = true
        end
    end

    for n in 1:m.num_exp
        if add_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], dual_conic[m.r_idx_exp_subp[n]], dual_conic[m.s_idx_exp_subp[n]])
            is_viol_subp = true
        end
    end

    # for n in 1:m.num_sdp
    #
    #
    #     add_cut_sdp!(m, m.vars_sdp[n], dual[m.rows_sdp[n]], m.smat[n], m.logs[:SDP])
    # end

    return (false, is_viol_subp)
end

# Solve conic subproblem given some solution to the integer variables, update incumbent
function solve_subp!(m, b_sub_int::Vector{Float64})
    # Load/solve conic model
    tic()
    if m.update_conicsub
        # Reuse model already created by changing b vector
        MathProgBase.setbvec!(m.model_conic, b_sub_int)
    else
        # Load new model
        if m.dualize_sub
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end

        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, b_sub_int, m.cone_con_sub, m.cone_var_sub)
    end

    MathProgBase.optimize!(m.model_conic)
    m.logs[:n_conic] += 1
    m.logs[:conic_solve] += toq()

    status_conic = MathProgBase.status(m.model_conic)
    if status_conic == :Optimal
        m.logs[:n_opt] += 1
    elseif status_conic == :Infeasible
        m.logs[:n_inf] += 1
    elseif status_conic == :Suboptimal
        m.logs[:n_sub] += 1
    elseif status_conic == :UserLimit
        m.logs[:n_lim] += 1
    elseif status_conic == :ConicFailure
        m.logs[:n_fail] += 1
    else
        m.logs[:n_other] += 1
    end

    if (status_conic == :Optimal) || (status_conic == :Suboptimal)
        # Get solution
        soln_conic = MathProgBase.getsolution(m.model_conic)
    else
        soln_conic = Float64[]
    end

    if (status_conic == :Infeasible) || (status_conic == :Optimal) || (status_conic == :Suboptimal)
        # Get dual
        dual_conic = MathProgBase.getdual(m.model_conic)
    else
        dual_conic = Float64[]
    end

    # Free the conic model if not saving it
    if !m.update_conicsub && applicable(MathProgBase.freemodel!, m.model_conic)
        MathProgBase.freemodel!(m.model_conic)
    end

    return (status_conic, soln_conic, dual_conic)
end

# Check cone infeasibilities of current solution, optionally add K* cuts from current solution for infeasible cones
function add_prim_feas_cuts!(m, add_cuts::Bool)
    is_infeas = false
    is_viol_prim = false

    for n in 1:m.num_soc
        # Get cone current solution, check infeasibility
        v_vals = getvalue(m.v_soc[n])
        inf_outer = vecnorm(v_vals) - getvalue(m.t_soc[n])
        if inf_outer < m.tol_prim_infeas
            continue
        end
        is_infeas = true
        if !add_cuts
            continue
        end

        # Add SOC K* primal cuts from solution
        # Dual is (1, -1/norm(v)*v)
        if add_cut_soc!(m, m.t_soc[n], m.v_soc[n], m.d_soc[n], m.a_soc[n], -1/vecnorm(v_vals)*v_vals)
            is_viol_prim = true
        end
    end

    for n in 1:m.num_exp
        r_val = getvalue(m.r_exp[n])
        s_val = getvalue(m.s_exp[n])
        inf_outer = s_val*exp(r_val/s_val) - getvalue(m.t_exp[n])
        if inf_outer < m.tol_prim_infeas
            continue
        end
        is_infeas = true
        if !add_cuts
            continue
        end

        # Add ExpPrimal K* primal cuts from solution
        # Dual is (-exp(r/s), exp(r/s)(r/s-1), 1)
        ers = exp(r_val/s_val)
        if add_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], -ers, ers*(r_val/s_val-1))
            is_viol_prim = true
        end
    end

    # for n in 1:m.num_sdp
    #
    #
    #     add_cut_sdp!(m, m.dim_sdp[n], m.vars_sdp[n], dual[m.rows_sdp[n]], m.smat[n])
    # end

    return (is_infeas, is_viol_prim)
end

# Remove near-zeros from a vector, return false if all values are near-zeros
function clean_zeros!(m, data::Vector{Float64})
    keep = false
    for j in 1:length(data)
        if abs(data[j]) < m.tol_zero
            data[j] = 0.
        else
            keep = true
        end
    end
    return keep
end

# Add K* cuts for a SOC, return true if a cut is violated by current solution
function add_cut_soc!(m, t, v, d, a, v_dual)
    # Remove near-zeros, return false if all near zero
    if !clean_zeros!(m, v_dual)
        return false
    end

    # Calculate t_dual according to SOC definition
    t_dual = vecnorm(v_dual)

    dim = length(v)
    is_viol = false
    add_full = false
    if m.soc_disagg
        # Using SOC disaggregation
        for j in 1:dim
            if v_dual[j] == 0.
                # Zero cut, don't add
                continue
            elseif dim*v_dual[j]^2/t_dual < m.tol_zero
                # Coefficient is too small, don't add, add full cut later
                add_full = true
                continue
            elseif (v_dual[j]/t_dual)^2 < 1e-5
                # Cut is poorly conditioned, add it but also add full cut
                add_full = true
            end

            # TODO is the coeff on a or v supposed to be negative?
            if m.soc_abslift
                # Using SOC absvalue lifting, so add two-sided cut
                # (v'_j)^2/norm(v')*t + 2*norm(v')*d_j - 2*|v'_j|*a_j >= 0
                # Scale by 2*dim
                @expression(m.model_mip, cut_expr, 2*dim*(v_dual[j]^2/t_dual*t + 2*t_dual*d[j] - 2*abs(v_dual[j])*a[j]))
            else
                # Not using SOC absvalue lifting, so add a single one-sided cut
                # (v'_j)^2/norm(v')*t + 2*norm(v')*d_j + 2*v'_j*v_j >= 0
                # Scale by 2*dim
                @expression(m.model_mip, cut_expr, 2*dim*(v_dual[j]^2/t_dual*t + 2*t_dual*d[j] + 2*v_dual[j]*v[j]))
            end
            if add_cut!(m, cut_expr, m.logs[:SOC])
                is_viol = true
            end
        end
    end

    if add_full || !m.soc_disagg
        # Using full SOC cut
        if m.soc_abslift
            # Using SOC absvalue lifting, so add many-sided cut
            # norm(v')*t - dot(|v'|,a) >= 0
            # Scale by 2
            @expression(m.model_mip, cut_expr, 2*(t_dual*t - vecdot(abs(v_dual), a)))
        else
            # Not using SOC absvalue lifting, so add a single one-sided cut
            # norm(v')*t + dot(v',v) >= 0
            @expression(m.model_mip, cut_expr, t_dual*t + vecdot(v_dual, v))
        end
        if add_cut!(m, cut_expr, m.logs[:SOC])
            is_viol = true
        end
    end

    return is_viol
end

# Add K* cuts for a ExpPrimal, return true if a cut is violated by current solution
function add_cut_exp!(m, r, s, t, r_dual, s_dual)
    # Clean zeros
    if r_dual >= -m.tol_zero
        return false
    end
    if abs(s_dual) < m.tol_zero
        s_dual = 0.
    end

    # Calculate t_dual according to dual exp cone definition
    # (u,v,w) in ExpDual <-> exp(1)*w >= -u*exp(v/u), w >= 0, u < 0
    t_dual = -r_dual*exp(s_dual/r_dual - 1)
    if t_dual < m.tol_zero
        return false
    end

    # Cut is (u,v,w)'(r,s,t) >= 0
    @expression(m.model_mip, cut_expr, r_dual*r + s_dual*s + t_dual*t)

    return add_cut!(m, cut_expr, m.logs[:ExpPrimal])
end

# Add K* cuts for a PSD, return true if a cut is violated by current solution
function add_cut_sdp!(m, dim, vars, cut, smat, is_viol, summary)
    nothing
end

# Check and record violation and add cut, return true if violated
function add_cut!(m, cut_expr::JuMP.AffExpr, cone_logs::Dict{Symbol,Any})
    if !m.oa_started
        @constraint(m.model_mip, cut_expr >= 0)
        return false
    end

    cut_inf = -getvalue(cut_expr)

    if cut_inf > m.tol_prim_infeas
        if m.mip_solver_drives
            @lazyconstraint(m.cb_lazy, cut_expr >= 0)
        else
            @constraint(m.model_mip, cut_expr >= 0)
        end

        if m.log_level > 2
            cone_logs[:n_viol] += 1
            cone_logs[:viol_max] = max(cut_inf, cone_logs[:viol_max])
        end
        return true
    elseif !m.viol_cuts_only
        if m.mip_solver_drives
            @lazyconstraint(m.cb_lazy, cut_expr >= 0)
        else
            @constraint(m.model_mip, cut_expr >= 0)
        end

        if m.log_level > 2
            cone_logs[:n_nonviol] += 1
            cone_logs[:nonviol_max] = max(-cut_inf, cone_logs[:nonviol_max])
        end
        return false
    end

    return false
end


#=========================================================
 Logging and printing functions
=========================================================#

# Create dictionary of logs for timing and iteration counts
function create_logs!(m)
    logs = Dict{Symbol,Any}()

    # Timers
    logs[:total] = 0.       # Performing total optimize algorithm
    logs[:data_trans] = 0.  # Transforming data
    logs[:data_conic] = 0.  # Generating conic data
    logs[:data_mip] = 0.    # Generating MIP data
    logs[:relax_solve] = 0. # Solving initial conic relaxation model
    logs[:oa_alg] = 0.      # Performing outer approximation algorithm
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:conic_solve] = 0. # Solving conic subproblem model

    # Counters
    logs[:n_mip] = 0        # Number of times lazy is called
    logs[:n_lazy] = 0       # Number of times lazy is called
    logs[:n_feas] = 0       # Number of times lazy is called with feasible solution
    logs[:n_repeat] = 0     # Number of times integer solution repeats
    logs[:n_conic] = 0      # Number of unique integer solutions (conic subproblem solves)
    logs[:n_nodual] = 0     # Number of times no violated dual cuts could be added in lazy
    logs[:n_nocuts] = 0     # Number of times no violated cuts could be added on infeas solution in lazy

    logs[:n_heur] = 0       # Number of times heuristic is called
    logs[:n_add] = 0        # Number of times heuristic adds new solution

    logs[:n_incum] = 0      # Number of times incumbent is called
    logs[:n_innew] = 0      # Number of times incumbent allows feas soln

    logs[:n_inf] = 0        # Number of conic subproblem infeasible statuses
    logs[:n_opt] = 0        # Number of conic subproblem optimal statuses
    logs[:n_sub] = 0        # Number of conic subproblem suboptimal statuses
    logs[:n_lim] = 0        # Number of conic subproblem user limit statuses
    logs[:n_fail] = 0       # Number of conic subproblem conic failure statuses
    logs[:n_other] = 0      # Number of conic subproblem other statuses

    # logs[:n_relax] = 0      # Number of relaxation subproblem cuts added
    # logs[:n_dualfullv] = 0  # Number of violated full subproblem cuts added (after relax)
    # logs[:n_dualfullnv] = 0 # Number of nonviolated full subproblem cuts added (after relax)
    # logs[:n_dualdisv] = 0   # Number of violated disagg subproblem cuts added (after relax)
    # logs[:n_dualdisnv] = 0  # Number of nonviolated disagg subproblem cuts added (after relax)
    # logs[:n_dualdiscon] = 0 # Number of poorly conditioned disagg subproblem cuts encountered
    # logs[:n_dualrev] = 0    # Number of violated full subproblem cuts RE-added (integer solution repeated)
    # logs[:n_primfullv] = 0  # Number of violated full prim cuts added
    # logs[:n_primfullnv] = 0 # Number of nonviolated full prim cuts added
    # logs[:n_primdisv] = 0   # Number of violated disagg prim cuts added
    # logs[:n_primdisnv] = 0  # Number of nonviolated disagg prim cuts added
    # logs[:n_primdiscon] = 0 # Number of poorly conditioned disagg primal cuts encountered

    logs_soc = Dict{Symbol,Any}()
    logs_soc[:relax] = 0
    logs_soc[:n_viol_total] = 0
    logs_soc[:n_nonviol_total] = 0
    logs[:SOC] = logs_soc

    logs_exp = Dict{Symbol,Any}()
    logs_exp[:relax] = 0
    logs_exp[:n_viol_total] = 0
    logs_exp[:n_nonviol_total] = 0
    logs[:ExpPrimal] = logs_exp

    logs_sdp = Dict{Symbol,Any}()
    logs_sdp[:relax] = 0
    logs_sdp[:n_viol_total] = 0
    logs_sdp[:n_nonviol_total] = 0
    logs[:SDP] = logs_sdp

    m.logs = logs
end

# Reset all cone cut summary info to 0
function reset_cone_logs!(m)
    if m.log_level <= 2
        return
    end

    if m.num_soc > 0
        logs_soc = m.logs[:SOC]
        logs_soc[:n_viol_total] += logs_soc[:n_viol]
        logs_soc[:n_nonviol_total] += logs_soc[:n_nonviol]
        logs_soc[:n_viol] = 0
        logs_soc[:viol_max] = 0.
        logs_soc[:n_nonviol] = 0
        logs_soc[:nonviol_max] = 0.
    end

    if m.num_exp > 0
        logs_exp = m.logs[:ExpPrimal]
        logs_exp[:n_viol_total] += logs_exp[:n_viol]
        logs_exp[:n_nonviol_total] += logs_exp[:n_nonviol]
        logs_exp[:n_viol] = 0
        logs_exp[:viol_max] = 0.
        logs_exp[:n_nonviol] = 0
        logs_exp[:nonviol_max] = 0.
    end

    if m.num_sdp > 0
        logs_sdp = m.logs[:SDP]
        logs_sdp[:n_viol_total] += logs_sdp[:n_viol]
        logs_sdp[:n_nonviol_total] += logs_sdp[:n_nonviol]
        logs_sdp[:n_viol] = 0
        logs_sdp[:viol_max] = 0.
        logs_sdp[:n_nonviol] = 0
        logs_sdp[:nonviol_max] = 0.
    end
end

# Print objective gap information for iterative
function print_gap(m)
    if m.log_level <= 1
        return
    end

    if (m.logs[:n_mip] == 1) || (m.log_level > 2)
        @printf "\n%-4s | %-14s | %-14s | %-11s | %-11s\n" "Iter" "Best obj" "OA obj" "Rel gap" "Time (s)"
    end
    if m.gap_rel_opt < 1000
        @printf "%4d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" m.logs[:n_mip] m.best_obj m.mip_obj m.gap_rel_opt (time() - m.logs[:oa_alg])
    elseif isnan(m.gap_rel_opt)
        @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.logs[:n_mip] m.best_obj m.mip_obj "Inf" (time() - m.logs[:oa_alg])
    else
        @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.logs[:n_mip] m.best_obj m.mip_obj ">1000" (time() - m.logs[:oa_alg])
    end
    flush(STDOUT)
end

# Print after finish
function print_finish(m::PajaritoConicModel)
    flush(STDOUT)

    if m.log_level < 0
        @printf "\n"
        return
    end

    @printf "\nPajarito MICP solve summary:\n"

    @printf " - Total time (s)       = %14.2e\n" m.logs[:total]
    @printf " - Status               = %14s\n" m.status
    @printf " - Best feasible obj.   = %+14.6e\n" m.best_obj
    @printf " - OA obj. bound        = %+14.6e\n" m.mip_obj
    @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt

    flush(STDOUT)

    if m.log_level == 0
        @printf "\n"
        return
    end

    @printf "\nTimers (s):\n"

    @printf " - Setup                = %10.2e\n" (m.logs[:total] - m.logs[:oa_alg])
    @printf " -- Transform data      = %10.2e\n" m.logs[:data_trans]
    @printf " -- Create conic data   = %10.2e\n" m.logs[:data_conic]
    @printf " -- Create MIP data     = %10.2e\n" m.logs[:data_mip]
    @printf " -- Load/solve relax    = %10.2e\n" m.logs[:relax_solve]

    @printf " - MIP-driven algorithm = %10.2e\n" m.logs[:oa_alg]
    @printf " -- Solve conic model   = %10.2e\n" m.logs[:conic_solve]

    @printf "\nCounters:\n"

    @printf " - Lazy callback        = %5d\n" m.logs[:n_lazy]
    @printf " -- Feasible soln       = %5d\n" m.logs[:n_feas]
    @printf " -- Integer repeat      = %5d\n" m.logs[:n_repeat]
    @printf " -- Conic statuses      = %5d\n" m.logs[:n_conic]
    @printf " --- Infeasible         = %5d\n" m.logs[:n_inf]
    @printf " --- Optimal            = %5d\n" m.logs[:n_opt]
    @printf " --- Suboptimal         = %5d\n" m.logs[:n_sub]
    @printf " --- UserLimit          = %5d\n" m.logs[:n_lim]
    @printf " --- ConicFailure       = %5d\n" m.logs[:n_fail]
    @printf " --- Other status       = %5d\n" m.logs[:n_other]
    @printf " -- No viol. subp. cut  = %5d\n" m.logs[:n_nodual]
    @printf " -- No viol. prim. cut  = %5d\n" m.logs[:n_nocuts]

    @printf " - Heuristic callback   = %5d\n" m.logs[:n_heur]
    @printf " -- Feasible added      = %5d\n" m.logs[:n_add]

    @printf " - Incumbent callback   = %5d\n" m.logs[:n_incum]
    @printf " -- Feasible accepted   = %5d\n" m.logs[:n_innew]

    # @printf " - Subproblem cuts            = %5d\n" (m.logs[:n_relax] + m.logs[:n_dualfullv] + m.logs[:n_dualfullnv] + m.logs[:n_dualdisv] + m.logs[:n_dualdisnv])
    # @printf " -- Relaxation          = %5d\n" m.logs[:n_relax]
    # @printf " -- Full viol.          = %5d\n" m.logs[:n_dualfullv]
    # @printf " -- Full nonviol.       = %5d\n" m.logs[:n_dualfullnv]
    # @printf " -- Disagg. viol        = %5d\n" m.logs[:n_dualdisv]
    # @printf " -- Disagg. nonviol     = %5d\n" m.logs[:n_dualdisnv]
    # @printf " -- Poorly conditioned  = %5d\n" m.logs[:n_dualdiscon]
    #
    # @printf " - Primal cuts          = %5d\n" (m.logs[:n_primfullv] + m.logs[:n_primfullnv] + m.logs[:n_primdisv] + m.logs[:n_primdisnv])
    # @printf " -- Full viol.          = %5d\n" m.logs[:n_primfullv]
    # @printf " -- Full nonviol.       = %5d\n" m.logs[:n_primfullnv]
    # @printf " -- Disagg. viol        = %5d\n" m.logs[:n_primdisv]
    # @printf " -- Disagg. nonviol     = %5d\n" m.logs[:n_primdisnv]
    # @printf " -- Poorly conditioned  = %5d\n" m.logs[:n_primdiscon]

    flush(STDOUT)

    if !isfinite(m.best_obj) || any(isnan(m.final_soln))
        @printf "\n"
        return
    end

    @printf "\nMax absolute primal infeasibilities\n"

    # Constraint cones
    viol_lin = 0.0
    viol_soc = 0.0
    viol_rot = 0.0
    viol_exp = 0.0
    viol_sdp = 0.0
    vals = m.b_orig - m.A_orig * m.final_soln

    for (cone, idx) in m.cone_con_orig
        if cone == :Free
            nothing
        elseif cone == :Zero
            viol_lin = max(viol_lin, maxabs(vals[idx]))
        elseif cone == :NonNeg
            viol_lin = max(viol_lin, -minimum(vals[idx]))
        elseif cone == :NonPos
            viol_lin = max(viol_lin, maximum(vals[idx]))
        elseif cone == :SOC
            viol_soc = max(viol_soc, vecnorm(vals[idx[j]] for j in 2:length(idx)) - vals[idx[1]])
        elseif cone == :SOCRotated
            # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt2*x) in SOC, y >= 0, z >= 0
            viol_rot = max(viol_rot, sqrt((vals[idx[1]] - vals[idx[2]])^2 + 2. * sumabs2(vals[idx[j]] for j in 3:length(idx))) - (vals[idx[1]] + vals[idx[2]]))
        else
            error("Cone not supported: $cone\n")
        end
    end

    @printf " - Constraint cones:\n"
    @printf " -- Linear              = %10.2e\n" viol_lin
    @printf " -- Second order        = %10.2e\n" viol_soc
    @printf " -- Rot. second order   = %10.2e\n" viol_rot
    @printf " -- Primal exponential  = %10.2e\n" viol_exp
    @printf " -- Positive semidef.   = %10.2e\n" viol_sdp

    # Variable cones
    viol_lin = 0.0
    viol_soc = 0.0
    viol_rot = 0.0
    viol_exp = 0.0
    viol_sdp = 0.0
    vals = m.final_soln

    for (cone, idx) in m.cone_var_orig
        if cone == :Free
            nothing
        elseif cone == :Zero
            viol_lin = max(viol_lin, maxabs(vals[idx]))
        elseif cone == :NonNeg
            viol_lin = max(viol_lin, -minimum(vals[idx]))
        elseif cone == :NonPos
            viol_lin = max(viol_lin, maximum(vals[idx]))
        elseif cone == :SOC
            viol_soc = max(viol_soc, vecnorm(vals[idx[j]] for j in 2:length(idx)) - vals[idx[1]])
        elseif cone == :SOCRotated
            # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt2*x) in SOC, y >= 0, z >= 0
            viol_rot = max(viol_rot, sqrt((vals[idx[1]] - vals[idx[2]])^2 + 2. * sumabs2(vals[idx[j]] for j in 3:length(idx))) - (vals[idx[1]] + vals[idx[2]]))
        else
            error("Cone not supported: $cone\n")
        end
    end

    @printf " - Variable cones:\n"
    @printf " -- Linear              = %10.2e\n" viol_lin
    @printf " -- Second order        = %10.2e\n" viol_soc
    @printf " -- Rot. second order   = %10.2e\n" viol_rot
    @printf " -- Primal exponential  = %10.2e\n" viol_exp
    @printf " -- Positive semidef.   = %10.2e\n" viol_sdp

    @printf "\n"
    flush(STDOUT)
end
