#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
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
=========================================================#

using JuMP
using ConicBenchmarkUtilities


#=========================================================
 Constants
=========================================================#

const sqrt2 = sqrt(2)
const sqrt2inv = 1/sqrt2

const infeas_ray_tol = 1e-10 # For checking if conic subproblem infeasible ray is sufficiently negative

const feas_factor = 100. # For checking if solution is considered feasible from its maximum conic violation

const unstable_soc_disagg_tol = 1e-5 # For checking if a disaggregated SOC cut is numerically unstable


#=========================================================
 Conic model object
=========================================================#

mutable struct PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
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
    soc_disagg::Bool            # Disaggregate SOC using extended formulation
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
    sep_cuts_only::Bool         # Add primal cuts, do not add subproblem cuts
    sep_cuts_always::Bool       # Add primal cuts and subproblem cuts
    sep_cuts_assist::Bool       # Add subproblem cuts, and add primal cuts only subproblem cuts cannot be added

    cut_zero_tol::Float64       # Zero tolerance for cut coefficients
    mip_feas_tol::Float64       # Absolute feasibility tolerance used for primal cuts (set equal to feasibility tolerance of `mip_solver`)

    dump_subproblems::Bool      # Save each conic subproblem in conic benchmark format (CBF) at a specified location
    dump_basename::String       # Basename of conic subproblem CBF files: "/path/to/foo" creates files "/path/to/foo_NN.cbf" where "NN" is a counter

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
    num_cones::Int              # Number of cones in the MIP (linear and non-linear)

    # SOC data
    num_soc::Int                # Number of SOCs
    r_idx_soc_subp::Vector{Int} # Row index of r variable in SOCs in subproblems
    t_idx_soc_subp::Vector{Vector{Int}} # Row indices of t variables in SOCs in subproblem
    r_soc::Vector{JuMP.AffExpr} # r variable (epigraph) in SOCs
    t_soc::Vector{Vector{JuMP.AffExpr}} # t variables in SOCs
    pi_soc::Vector{Vector{JuMP.Variable}} # pi variables (disaggregated) in SOCs
    rho_soc::Vector{Vector{JuMP.Variable}} # rho variables (absolute values) in SOCs

    # ExpPrimal data
    num_exp::Int                # Number of ExpPrimal cones
    r_idx_exp_subp::Vector{Int} # Row index of r variable in ExpPrimals in subproblems
    s_idx_exp_subp::Vector{Int} # Row index of s variable in ExpPrimals in subproblems
    t_idx_exp_subp::Vector{Int} # Row index of t variable in ExpPrimals in subproblems
    r_exp::Vector{JuMP.AffExpr} # r variable in ExpPrimals
    s_exp::Vector{JuMP.AffExpr} # s variable in ExpPrimals
    t_exp::Vector{JuMP.AffExpr} # t variable in ExpPrimals

    # SDP data
    num_sdp::Int                # Number of SDP cones
    t_idx_sdp_subp::Vector{Vector{Int}} # Row indices of svec v variables in SDPs in subproblem
    smat_sdp::Vector{Symmetric{Float64,Array{Float64,2}}} # Preallocated array for smat space values
    T_sdp::Vector{Array{JuMP.AffExpr,2}} # smat space T variables in SDPs

    # Miscellaneous for algorithms
    inf_subp_scale::Float64     # Calculated infeasible subproblem cuts scaling factor
    opt_subp_scale::Float64     # Calculated optimal subproblem cuts scaling factor
    update_conicsub::Bool       # Indicates whether to use setbvec! to update an existing conic subproblem model
    model_conic::MathProgBase.AbstractConicModel # Conic subproblem model: persists when the conic solver implements MathProgBase.setbvec!
    oa_started::Bool            # Indicator for Iterative or MIP-solver-driven algorithms started
    cache_dual::Dict{Vector{Float64},Vector{Float64}} # Set of integer solution subvectors already seen
    new_incumb::Bool            # Indicates whether a new incumbent solution from the conic solver is waiting to be added as warm-start or heuristic
    cb_heur                     # Heuristic callback reference (MIP-driven only)
    cb_lazy                     # Lazy callback reference (MIP-driven only)
    aggregate_cut::JuMP.AffExpr # If not disaggregating cuts on nonpolyhedral cones, build up single cut expression here before adding

    # Solution and bound information
    is_best_conic::Bool         # Indicates best feasible came from conic solver solution, otherwise MIP solver solution
    best_bound::Float64         # Best lower bound from MIP
    best_obj::Float64           # Best feasible objective value
    best_int::Vector{Float64}   # Best feasible integer solution
    best_cont::Vector{Float64}  # Best feasible continuous solution
    gap_rel_opt::Float64        # Relative optimality gap = |best_bound - best_obj|/|best_obj|
    final_soln::Vector{Float64} # Final solution on original variables

    # Logging information and status
    logs::Dict{Symbol,Any}      # Logging information
    status::Symbol              # Current Pajarito status

    # Model constructor
    function PajaritoConicModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, mip_subopt_solver, mip_subopt_count, round_mip_sols, use_mip_starts, cont_solver, solve_relax, solve_subp, dualize_relax, dualize_subp, all_disagg, soc_disagg, soc_abslift, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, scale_subp_cuts, scale_subp_factor, scale_subp_up, viol_cuts_only, sep_cuts_only, sep_cuts_always, sep_cuts_assist, cut_zero_tol, mip_feas_tol, dump_subproblems, dump_basename)
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.solve_relax = solve_relax
        m.solve_subp = solve_subp
        m.dualize_relax = dualize_relax
        m.dualize_subp = dualize_subp
        m.use_mip_starts = use_mip_starts
        m.round_mip_sols = round_mip_sols
        m.mip_subopt_count = mip_subopt_count
        m.mip_subopt_solver = mip_subopt_solver
        m.soc_in_mip = soc_in_mip
        m.all_disagg = all_disagg
        m.soc_disagg = soc_disagg
        m.soc_abslift = soc_abslift
        m.init_soc_one = init_soc_one
        m.init_soc_inf = init_soc_inf
        m.init_exp = init_exp
        m.scale_subp_cuts = scale_subp_cuts
        m.scale_subp_factor = scale_subp_factor
        m.scale_subp_up = scale_subp_up
        m.viol_cuts_only = viol_cuts_only
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.cut_zero_tol = cut_zero_tol
        m.sep_cuts_only = sep_cuts_only
        m.sep_cuts_always = sep_cuts_always
        m.sep_cuts_assist = sep_cuts_assist
        m.mip_feas_tol = mip_feas_tol
        m.init_sdp_lin = init_sdp_lin
        m.init_sdp_soc = init_sdp_soc
        m.sdp_eig = sdp_eig
        m.sdp_soc = sdp_soc
        m.dump_subproblems = dump_subproblems
        m.dump_basename = dump_basename

        m.var_types = Symbol[]
        # m.var_start = Float64[]
        m.num_var_orig = 0
        m.num_con_orig = 0

        m.oa_started = false
        m.best_obj = Inf
        m.best_bound = -Inf
        m.gap_rel_opt = NaN
        m.status = :NotLoaded

        create_logs!(m)

        return m
    end
end


#=========================================================
 MathProgBase functions
=========================================================#

# Verify initial conic data and convert appropriate types and store in Pajarito model
function MathProgBase.loadproblem!(m::PajaritoConicModel, c, A, b, cone_con, cone_var)
    # Check dimensions of conic problem
    num_con_orig = length(b)
    num_var_orig = length(c)
    if size(A) != (num_con_orig, num_var_orig)
        error("Dimension mismatch between A matrix $(size(A)), b vector ($(length(b))), and c vector ($(length(c)))\n")
    end
    if isempty(cone_con) || isempty(cone_var)
        error("Variable or constraint cones are missing\n")
    end

    A_sp = sparse(A)
    dropzeros!(A_sp)

    if m.log_level > 1
        @printf "\nProblem dimensions:\n"
        @printf "%16s | %7d\n" "variables" num_var_orig
        @printf "%16s | %7d\n" "constraints" num_con_orig
        @printf "%16s | %7d\n" "nonzeros in A" nnz(A_sp)
    end

    # Check constraint cones
    inds_con = zeros(Int, num_con_orig)
    for (spec, inds) in cone_con
        if spec == :Free
            error("A cone $spec should not be in the constraint cones\n")
        end

        if any(inds .> num_con_orig)
            error("Some indices in a constraint cone do not correspond to indices of vector b\n")
        end

        for i in inds
            inds_con[i] += 1
        end
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

        for i in inds
            inds_var[i] += 1
        end
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

    if m.log_level > 1
        @printf "\nCones summary:"
        @printf "\n%-16s | %-7s | %-9s | %-9s\n" "Cone" "Count" "Min dim." "Max dim."
        if num_soc > 0
            @printf "%16s | %7d | %9d | %9d\n" "Second order" num_soc min_soc max_soc
        end
        if num_rot > 0
            @printf "%16s | %7d | %9d | %9d\n" "Rotated S.O." num_rot min_rot max_rot
        end
        if num_exp > 0
            @printf "%16s | %7d | %9d | %9d\n" "Primal expon." num_exp 3 3
        end
        if num_sdp > 0
            min_side = round(Int, sqrt(1/4+2*min_sdp)-1/2)
            max_side = round(Int, sqrt(1/4+2*max_sdp)-1/2)
            @printf "%16s | %7d | %7s^2 | %7s^2\n" "Pos. semidef." num_sdp min_side max_side
        end
    end

    if m.solve_relax || m.solve_subp
        # Verify cone compatibility with conic solver
        conic_spec = MathProgBase.supportedcones(m.cont_solver)

        # Pajarito converts rotated SOCs to standard SOCs
        if :SOC in conic_spec
            push!(conic_spec, :SOCRotated)
        end

        # Error if a cone in data is not supported
        for (spec, _) in vcat(cone_con, cone_var)
            if !(spec in conic_spec)
                error("Cones $spec are not supported by the specified conic solver (only $conic_spec)\n")
            end
        end
    end

    # Save original data
    m.num_con_orig = length(b)
    m.num_var_orig = length(c)
    m.c_orig = c
    m.A_orig = A_sp
    m.b_orig = b
    m.cone_con_orig = cone_con
    m.cone_var_orig = cone_var

    m.final_soln = fill(NaN, m.num_var_orig)
    m.status = :Loaded
    flush(stdout)
    flush(stderr)
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
        error("Must call setvartype! immediately after loadproblem!\n")
    end
    if length(var_types) != m.num_var_orig
        error("Variable types vector length ($(length(var_types))) does not match number of variables ($(m.num_var_orig))\n")
    end

    num_cont = 0
    num_bin = 0
    num_int = 0
    for vtype in var_types
        if vtype == :Cont
            num_cont += 1
        elseif vtype == :Bin
            num_bin += 1
        elseif vtype == :Int
            num_int += 1
        else
            error("A variable type ($vtype) is invalid; variable types must be :Bin, :Int, or :Cont\n")
        end
    end

    if (num_bin + num_int) == 0
        error("No variable types are :Bin or :Int; use the continuous conic solver directly if your problem is continuous\n")
    end

    if m.log_level > 1
        @printf "\nVariable types:\n"
        if num_cont > 0
            @printf "%16s | %7d\n" "continuous" num_cont
        end
        if num_bin > 0
            @printf "%16s | %7d\n" "binary" num_bin
        end
        if num_int > 0
            @printf "%16s | %7d\n" "integer" num_int
        end
    end

    m.var_types = var_types
    flush(stdout)
    flush(stderr)
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    if m.status != :Loaded
        error("Must call optimize! function after setvartype! and loadproblem!\n")
    end
    if isempty(m.var_types)
        error("Variable types were not specified (use setvartype! function)\n")
    end

    m.logs[:total] = time()

    # Transform data
    if m.log_level > 1
        @printf "\n%-33s" "Transforming data..."
    end
    start_time_trans = time()
    (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new, cols_cont, cols_int) = transform_data(copy(m.c_orig), copy(m.A_orig), copy(m.b_orig), deepcopy(m.cone_con_orig), deepcopy(m.cone_var_orig), copy(m.var_types), m.solve_relax)
    m.logs[:data_trans] += time() - start_time_trans
    if m.log_level > 1
        @printf "%6.2fs\n" m.logs[:data_trans]
    end

    if m.solve_subp
        # Create conic subproblem
        if m.log_level > 1
            @printf "\n%-33s" "Creating conic subproblem..."
        end
        start_time_subp = time()

        map_rows_subp = create_conicsub_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, cols_cont, cols_int)

        if m.dualize_subp
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end
        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        if hasmethod(MathProgBase.setbvec!, (typeof(m.model_conic), Vector{Float64}))
            # Can use setbvec! on the conic subproblem model: load it
            m.update_conicsub = true
            MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, m.b_sub, m.cone_con_sub, m.cone_var_sub)
        else
            m.update_conicsub = false
        end

        m.logs[:data_conic] += time() - start_time_subp
        if m.log_level > 1
            @printf "%6.2fs\n" m.logs[:data_conic]
        end
    else
        map_rows_subp = zeros(Int, length(b_new))
        m.c_sub_cont = c_new[cols_cont]
        m.c_sub_int = c_new[cols_int]
    end

    # Create MIP model
    if m.log_level > 1
        @printf "\n%-33s" "Building MIP model..."
    end
    start_time_mip = time()
    (r_idx_soc_relx, t_idx_soc_relx, r_idx_exp_relx, s_idx_exp_relx, t_idx_exp_relx, t_idx_sdp_relx) = create_mip_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, map_rows_subp, cols_cont, cols_int)
    m.logs[:data_mip] += time() - start_time_mip
    if m.log_level > 1
        @printf "%6.2fs\n" m.logs[:data_mip]
    end
    flush(stdout)
    flush(stderr)

    # Calculate infeasible and optimal subproblem K* cuts scaling factors
    m.inf_subp_scale = m.mip_feas_tol*m.scale_subp_factor
    m.opt_subp_scale = m.mip_feas_tol/m.rel_gap*m.scale_subp_factor
    if m.all_disagg
        m.inf_subp_scale *= m.num_cones
        m.opt_subp_scale *= m.num_cones
    end

    if m.solve_relax
        # Solve relaxed conic problem, proceed with algorithm if optimal or suboptimal, else finish
        if m.log_level > 1
            @printf "\n%-33s" "Solving conic relaxation..."
        end
        start_time_relax = time()
        if m.dualize_relax
            solver_relax = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_relax = m.cont_solver
        end
        model_relax = MathProgBase.ConicModel(solver_relax)
        MathProgBase.loadproblem!(model_relax, c_new, A_new, b_new, cone_con_new, cone_var_new)
        MathProgBase.optimize!(model_relax)
        m.logs[:relax_solve] += time() - start_time_relax
        if m.log_level > 1
            @printf "%6.2fs\n" m.logs[:relax_solve]
        end

        status_relax = MathProgBase.status(model_relax)

        if status_relax == :Infeasible
            if m.log_level > 0
                println("Initial conic relaxation status was $status_relax\n")
            end
            m.status = :Infeasible
        elseif status_relax == :Unbounded
            @warn "Initial conic relaxation status was $status_relax\n"
            m.status = :UnboundedRelax
        else
            # if status_relax in (:Optimal, :Suboptimal, :PDFeas, :DualFeas)
            if m.log_level > 2
                @printf " - Relaxation status    = %14s\n" status_relax
            end

            dual_conic = Float64[]
            try
                dual_conic = MathProgBase.getdual(model_relax)
                if any(isnan, dual_conic)
                    dual_conic = Float64[]
                end
            catch
            end

            if !isempty(dual_conic) && !any(isnan, dual_conic)
                m.status = :SolvedRelax
                dual_obj = -dot(b_new, dual_conic)
                m.best_bound = dual_obj
                if m.log_level > 2
                    @printf " - Relaxation bound     = %14.6f\n" dual_obj
                end

                # Optionally scale dual
                if m.scale_subp_cuts
                    # Rescale by number of cones / absval of full conic objective
                    rmul!(dual_conic, m.opt_subp_scale/(abs(dual_obj) + 1e-5))
                end

                # Add relaxation cut(s)
                start_time_relax_cuts = time()

                m.aggregate_cut = JuMP.AffExpr(0)

                for n in 1:m.num_soc
                    u_val = dual_conic[r_idx_soc_relx[n]]
                    w_val = dual_conic[t_idx_soc_relx[n]]
                    add_subp_cut_soc!(m, m.r_soc[n], m.t_soc[n], m.pi_soc[n], m.rho_soc[n], u_val, w_val)
                end

                for n in 1:m.num_exp
                    u_val = dual_conic[r_idx_exp_relx[n]]
                    v_val = dual_conic[s_idx_exp_relx[n]]
                    w_val = dual_conic[t_idx_exp_relx[n]]
                    add_subp_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], u_val, v_val, w_val)
                end

                for n in 1:m.num_sdp
                    # Get smat space dual
                    W_val = make_smat!(m.smat_sdp[n], dual_conic[t_idx_sdp_relx[n]])
                    add_subp_cut_sdp!(m, m.T_sdp[n], W_val)
                end

                if !m.all_disagg
                    @constraint(m.model_mip, m.aggregate_cut >= 0)
                end

                m.logs[:relax_cuts] += time() - start_time_relax_cuts
            else
                m.status = :FailedRelax
            end
        end

        # Free the conic model
        if applicable(MathProgBase.freemodel!, model_relax)
            MathProgBase.freemodel!(model_relax)
        end
    end
    flush(stdout)
    flush(stderr)

    # Finish if exceeded timeout option, else proceed to MIP solves if not infeasible
    if (time() - m.logs[:total]) > m.timeout
        m.status = :UserLimit
    elseif m.status != :Infeasible
        # Initialize and begin iterative or MIP-solver-driven algorithm
        m.oa_started = true
        m.new_incumb = false
        m.cache_dual = Dict{Vector{Float64},Vector{Float64}}()

        if m.log_level >= 1
            @printf "\nStarting %s algorithm\n" (m.mip_solver_drives ? "MIP-solver-driven" : "iterative")
        end
        status_oa = m.mip_solver_drives ? solve_mip_driven!(m) : solve_iterative!(m)

        if status_oa == :Infeasible
            m.status = :Infeasible
        elseif status_oa == :Unbounded
            if !m.solve_relax
                @warn "MIP solver returned status $status_oa; try using the conic relaxation cuts (set solve_relax = true)\n"
            elseif m.status == :SolvedRelax
                @warn "MIP solver returned status $status_oa but the conic relaxation solve succeeded; try tightening the conic solver tolerances (or submit an issue)\n"
            else
                @warn "MIP solver returned status $status_oa and the conic relaxation solve failed; use a conic solver that succeeds on the relaxation (or submit an issue)\n"
            end
            m.status = :UnboundedOA
        elseif (status_oa == :UserLimit) || (status_oa == :Optimal) || (status_oa == :Suboptimal) || (status_oa == :FailedOA)
            if (status_oa == :Suboptimal) || (status_oa == :FailedOA)
                @warn "Pajarito failed to converge to the desired relative gap; try turning off the MIP solver's presolve functionality\n"
            end

            if isfinite(m.best_obj)
                # Have a best feasible solution, update final solution on original variables
                soln_new = zeros(length(c_new))
                soln_new[cols_int] = m.best_int
                soln_new[cols_cont] = m.best_cont
                m.final_soln = zeros(m.num_var_orig)
                m.final_soln[keep_cols] = soln_new
            end

            m.status = status_oa
        else
            @warn "MIP solver returned status $status_oa, which Pajarito does not handle\n"
            m.status = :FailedMIP
        end
    end
    flush(stdout)
    flush(stderr)

    # Finish timer and print summary
    m.logs[:total] = time() - m.logs[:total]
    print_finish(m)
    flush(stdout)
    flush(stderr)
end

MathProgBase.numconstr(m::PajaritoConicModel) = m.num_con_orig

MathProgBase.numvar(m::PajaritoConicModel) = m.num_var_orig

MathProgBase.status(m::PajaritoConicModel) = m.status

MathProgBase.getsolvetime(m::PajaritoConicModel) = m.logs[:total]

MathProgBase.getobjval(m::PajaritoConicModel) = m.best_obj

MathProgBase.getobjbound(m::PajaritoConicModel) = m.best_bound

MathProgBase.getsolution(m::PajaritoConicModel) = m.final_soln

function MathProgBase.getnodecount(m::PajaritoConicModel)
    if !m.mip_solver_drives
        error("Node count not defined when using iterative algorithm\n")
    else
        return MathProgBase.getnodecount(m.model_mip)
    end
end


#=========================================================
 Data and model functions
=========================================================#

# Transform/preprocess data
function transform_data(c_orig, A_orig, b_orig, cone_con_orig, cone_var_orig, var_types, solve_relax)
    (A_I, A_J, A_V) = findnz(A_orig)

    num_con_new = length(b_orig)
    b_new = b_orig
    cone_con_new = Tuple{Symbol,Vector{Int}}[(spec, vec(collect(inds))) for (spec, inds) in cone_con_orig]

    num_var_new = 0
    cone_var_new = Tuple{Symbol,Vector{Int}}[]

    old_new_col = zeros(Int, length(c_orig))

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

    keep_cols = findall(x->x!=0, old_new_col)
    c_new = c_orig[keep_cols]
    var_types_new = var_types[keep_cols]
    A_full = sparse(A_I, A_J, A_V, num_con_new, length(c_orig))
    A_keep = A_full[:, keep_cols]
    dropzeros!(A_keep)
    (A_I, A_J, A_V) = findnz(A_keep)

    # Convert SOCRotated cones to SOC cones (MathProgBase definitions)
    has_rsoc = false
    socr_rows = Vector{Int}[]
    for n_cone in 1:length(cone_con_new)
        (spec, rows) = cone_con_new[n_cone]
        if spec == :SOCRotated
            cone_con_new[n_cone] = (:SOC, rows)
            push!(socr_rows, rows)
            has_rsoc = true
        end
    end

    if has_rsoc
        row_to_nzind = map(t -> Int[], 1:num_con_new)
        for (ind, i) in enumerate(A_I)
            push!(row_to_nzind[i], ind)
        end

        for rows in socr_rows
            inds_1 = row_to_nzind[rows[1]]
            inds_2 = row_to_nzind[rows[2]]

            # Use old constraint cone SOCRotated for (sqrt2inv*(p1+p2),sqrt2inv*(-p1+p2),q) in SOC
            for ind in inds_1
                A_V[ind] *= sqrt2inv
            end
            for ind in inds_2
                A_V[ind] *= sqrt2inv
            end

            append!(A_I, fill(rows[1], length(inds_2)))
            append!(A_J, A_J[inds_2])
            append!(A_V, A_V[inds_2])

            append!(A_I, fill(rows[2], length(inds_1)))
            append!(A_J, A_J[inds_1])
            append!(A_V, -A_V[inds_1])

            b1 = b_new[rows[1]]
            b2 = b_new[rows[2]]
            b_new[rows[1]] = sqrt2inv*(b1 + b2)
            b_new[rows[2]] = sqrt2inv*(-b1 + b2)
        end
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

        bin_set_upper = falses(length(var_types_new))

        # For each bound-type constraint, tighten by rounding
        for (spec, rows) in cone_con_new
            if (spec != :NonNeg) && (spec != :NonPos)
                continue
            end

            for i in rows
                if row_slck_count[i] > 0
                    # Isolated variable x_j with b_i - a_ij*x_j in spec, b_i & a_ij nonzero
                    j = A_J[row_slck_count[i]]
                    type_j = var_types_new[j]
                    bound_j = b_new[i] / A_V[row_slck_count[i]]

                    if (spec == :NonNeg) && (A_V[row_slck_count[i]] > 0) || (spec == :NonPos) && (A_V[row_slck_count[i]] < 0)
                        # Upper bound: b_i/a_ij >= x_j
                        if type_j == :Bin
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
                        elseif type_j == :Int
                            # Tighten binary or integer upper bound by rounding down
                            if spec == :NonNeg
                                # floor >= x_j
                                b_new[i] = floor(bound_j)
                                A_V[row_slck_count[i]] = 1.
                            else
                                # -floor <= -x_j
                                b_new[i] = -floor(bound_j)
                                A_V[row_slck_count[i]] = -1.
                            end
                        end
                    else
                        # Lower bound: b_i/a_ij <= x_j
                        if type_j != :Cont
                            # Tighten binary or integer lower bound by rounding up
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
        for (j, j_type) in enumerate(var_types_new)
            if (j_type == :Bin) && !bin_set_upper[j]
                num_con_new += 1
                push!(A_I, num_con_new)
                push!(A_J, j)
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

    # Collect indices of continuous and integer variables
    cols_cont = findall(vt -> (vt == :Cont), var_types_new)
    cols_int = findall(vt -> (vt != :Cont), var_types_new)

    return (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new, cols_cont, cols_int)
end

# Create conic subproblem data
function create_conicsub_data!(m, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol}, cols_cont::Vector{Int}, cols_int::Vector{Int})
    # Build new subproblem variable cones by removing integer variables
    num_cont = 0
    cone_var_sub = Tuple{Symbol,Vector{Int}}[]

    for (spec, cols) in cone_var_new
        cols_cont_new = Int[]
        for j in cols
            if var_types_new[j] == :Cont
                num_cont += 1
                push!(cols_cont_new, num_cont)
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
    map_rows_subp = Vector{Int}(undef, num_con_new)

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
            map_rows_subp[rows] = collect((num_full + 1):(num_full + length(rows)))
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

    return map_rows_subp
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m, c_new::Vector{Float64}, A_new::SparseMatrixCSC{Float64,Int64}, b_new::Vector{Float64}, cone_con_new::Vector{Tuple{Symbol,Vector{Int}}}, cone_var_new::Vector{Tuple{Symbol,Vector{Int}}}, var_types_new::Vector{Symbol}, map_rows_subp::Vector{Int}, cols_cont::Vector{Int}, cols_int::Vector{Int})
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
                # setname(x_all[j], "x$(j)")
                setlowerbound(x_all[j], 0.)
            end
        elseif spec == :NonPos
            for j in cols
                # setname(x_all[j], "x$(j)")
                setupperbound(x_all[j], 0.)
            end
        # elseif spec == :Free
        #     for j in cols
        #         setname(x_all[j], "x$(j)")
        #     end
        end
    end

    # Allocate data for nonlinear cones
    # SOC data
    r_idx_soc_relx = Vector{Int}(undef, m.num_soc)
    t_idx_soc_relx = Vector{Vector{Int}}(undef, m.num_soc)
    r_idx_soc_subp = Vector{Int}(undef, m.num_soc)
    t_idx_soc_subp = Vector{Vector{Int}}(undef, m.num_soc)
    r_soc = Vector{JuMP.AffExpr}(undef, m.num_soc)
    t_soc = Vector{Vector{JuMP.AffExpr}}(undef, m.num_soc)
    pi_soc = Vector{Vector{JuMP.Variable}}(undef, m.num_soc)
    rho_soc = Vector{Vector{JuMP.Variable}}(undef, m.num_soc)

    # Exp data
    r_idx_exp_relx = Vector{Int}(undef, m.num_exp)
    s_idx_exp_relx = Vector{Int}(undef, m.num_exp)
    t_idx_exp_relx = Vector{Int}(undef, m.num_exp)
    r_idx_exp_subp = Vector{Int}(undef, m.num_exp)
    s_idx_exp_subp = Vector{Int}(undef, m.num_exp)
    t_idx_exp_subp = Vector{Int}(undef, m.num_exp)
    r_exp = Vector{JuMP.AffExpr}(undef, m.num_exp)
    s_exp = Vector{JuMP.AffExpr}(undef, m.num_exp)
    t_exp = Vector{JuMP.AffExpr}(undef, m.num_exp)

    # PSD data
    t_idx_sdp_relx = Vector{Vector{Int}}(undef, m.num_sdp)
    t_idx_sdp_subp = Vector{Vector{Int}}(undef, m.num_sdp)
    smat_sdp = Vector{Symmetric{Float64,Array{Float64,2}}}(undef, m.num_sdp)
    T_sdp = Vector{Array{JuMP.AffExpr,2}}(undef, m.num_sdp)

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    n_lin = 0
    n_soc = 0
    n_exp = 0
    n_sdp = 0

    @expression(model_mip, lhs_expr, b_new - A_new * x_all)

    for (spec, rows) in cone_con_new
        if spec == :NonNeg
            n_lin += length(rows)
            @constraint(model_mip, lhs_expr[rows] .>= 0)
        elseif spec == :NonPos
            n_lin += length(rows)
            @constraint(model_mip, lhs_expr[rows] .<= 0.)
        elseif spec == :Zero
            n_lin += length(rows)
            @constraint(model_mip, lhs_expr[rows] .== 0.)
        elseif spec == :SOC
            # Set up a SOC
            # (r,t) in SOC <-> r >= norm2(t) >= 0
            n_soc += 1

            if m.soc_in_mip
                # If putting SOCs in the MIP directly, don't need to use other SOC infrastructure
                @constraint(model_mip, lhs_expr[rows[1]] >= norm(lhs_expr[rows[2:end]]))

                r_idx_soc_relx[n_soc] = 0
                t_idx_soc_relx[n_soc] = Int[]
                r_idx_soc_subp[n_soc] = 0
                t_idx_soc_subp[n_soc] = Int[]
                continue
            end

            t_idx = rows[2:end]
            dim = length(t_idx)

            r_idx_soc_relx[n_soc] = rows[1]
            t_idx_soc_relx[n_soc] = t_idx
            r_idx_soc_subp[n_soc] = map_rows_subp[rows[1]]
            t_idx_soc_subp[n_soc] = map_rows_subp[t_idx]

            r_soc[n_soc] = r = lhs_expr[rows[1]]
            t_soc[n_soc] = t = lhs_expr[t_idx]

            if m.soc_disagg
                # Add disaggregated SOC variables pi
                # 2*pi_j >= t_j^2/r, all j
                pi = @variable(model_mip, [j in 1:dim], lowerbound=0)
                # for j in 1:dim
                #     setname(pi[j], "pi$(j)_soc$(n_soc)")
                # end

                # Add disaggregated SOC constraint
                # r - 2*sum(pi)
                # Scale by 2
                @constraint(model_mip, 2*r - 4*sum(pi) >= 0)
            else
                pi = Vector{JuMP.Variable}()
            end

            if m.soc_abslift
                # Add absolute value SOC variables rho
                # rho_j >= |t_j|
                rho = @variable(model_mip, [j in 1:dim], lowerbound=0)
                # for j in 1:dim
                #     setname(rho[j], "rho$(j)_soc$(n_soc)")
                # end

                # Add absolute value SOC constraints
                # rho_j >= t_j, rho_j >= -t_j
                # Scale by 2
                for j in 1:dim
                    @constraint(model_mip, 2*rho[j] - 2*t[j] >= 0)
                    @constraint(model_mip, 2*rho[j] + 2*t[j] >= 0)
                end
            else
                rho = Vector{JuMP.Variable}()
            end

            pi_soc[n_soc] = pi
            rho_soc[n_soc] = rho

            # Set bounds
            @constraint(model_mip, r >= 0)

            if m.init_soc_one
                # Add initial L_1 SOC linearizations if using disaggregation or absvalue lifting (otherwise no polynomial number of cuts)
                # r >= 1/sqrt(dim)*sum(|t_j|)
                if m.soc_disagg && m.soc_abslift
                    for j in 1:dim
                        # Disaggregated K* cut on (r, pi_j, rho_j) is (1/2*dim, 1, -1/sqrt(dim))
                        # Scale by 2*dim
                        @constraint(model_mip, r + 2*dim*pi[j] - 2*sqrt(dim)*rho[j] >= 0)
                    end
                elseif m.soc_disagg
                    for j in 1:dim
                        # Disaggregated K* cuts on (r, pi_j, t_j) are (1/2*dim, 1, (+/-)1/sqrt(dim))
                        # Scale by 2*dim
                        @constraint(model_mip, r + 2*dim*pi[j] - 2*sqrt(dim)*t[j] >= 0)
                        @constraint(model_mip, r + 2*dim*pi[j] + 2*sqrt(dim)*t[j] >= 0)
                    end
                elseif m.soc_abslift
                    # Non-disaggregated K* cut on (r, rho_1, ..., rho_dim) is (1, -1/sqrt(dim), ..., -1/sqrt(dim))
                    # Scale by 2
                    @constraint(model_mip, 2*r - 2/sqrt(dim)*sum(rho) >= 0)
                end
            end

            if m.init_soc_inf
                # Add initial L_inf SOC linearizations
                # r >= |t_j|, all j
                if m.soc_disagg && m.soc_abslift
                    for j in 1:dim
                        # Disaggregated K* cut on (r, pi_j, rho_j) is (1/2, 1, -1)
                        # Scale by 2*dim
                        @constraint(model_mip, dim*r + 2*dim*pi[j] - 2*dim*rho[j] >= 0)
                    end
                elseif m.soc_disagg
                    for j in 1:dim
                        # Disaggregated K* cuts on (r, pi_j, t_j) are (1/2, 1, (+/-) 1)
                        # Scale by 2*dim
                        @constraint(model_mip, dim*r + 2*dim*pi[j] - 2*dim*t[j] >= 0)
                        @constraint(model_mip, dim*r + 2*dim*pi[j] + 2*dim*t[j] >= 0)
                    end
                elseif m.soc_abslift
                    for j in 1:dim
                        # Non-disaggregated K* cut on (r, rho_1, ..., rho_j, ..., rho_dim) is (1, 0, ..., -1, ..., 0)
                        # Scale by 2*dim
                        @constraint(model_mip, 2*dim*r - 2*dim*rho[j] >= 0)
                    end
                else
                    for j in 1:dim
                        # Non-disaggregated K* cuts on (r, t_1, ..., t_j, ..., t_dim) are (1, 0, ..., (+/-) 1, ..., 0)
                        # Scale by dim
                        @constraint(model_mip, dim*r - dim*t[j] >= 0)
                        @constraint(model_mip, dim*r + dim*t[j] >= 0)
                    end
                end
            end
        elseif spec == :ExpPrimal
            # Set up a ExpPrimal cone
            # (t,s,r) in ExpPrimal <-> (s > 0 && r >= s*exp(t/s) > 0) || (s = 0 && r >= 0 && t <= 0)
            n_exp += 1

            r_idx_exp_relx[n_exp] = rows[3]
            s_idx_exp_relx[n_exp] = rows[2]
            t_idx_exp_relx[n_exp] = rows[1]
            r_idx_exp_subp[n_exp] = map_rows_subp[rows[3]]
            s_idx_exp_subp[n_exp] = map_rows_subp[rows[2]]
            t_idx_exp_subp[n_exp] = map_rows_subp[rows[1]]
            r_exp[n_exp] = r = lhs_expr[rows[3]]
            s_exp[n_exp] = s = lhs_expr[rows[2]]
            t_exp[n_exp] = t = lhs_expr[rows[1]]

            # Set bounds
            @constraint(model_mip, r >= 0)
            @constraint(model_mip, s >= 0)

            if m.init_exp
                # Add initial exp cuts using dual exp cone linearizations
                # (w,v,u) in ExpDual <-> (w < 0 && u >= -w*exp(v/w - 1) > 0) || (w = 0 && u >= 0 && v >= 0)
                # Add K* cut on (r,s,t) from points: w = -1; v in {-4, 1, 5}; u = exp(-v-1)
                for v in [-4, 1, 5]
                    @constraint(model_mip, exp(-v-1)*r + v*s + -t >= 0)
                end
            end
        elseif spec == :SDP
            # Set up a PSD cone
            # svec space is lowercase, smat space is uppercase
            # t in svec(PSD) <-> eigmin(smat(t)) >= 0
            # T in PSD <-> eigmin(T) >= 0
            n_sdp += 1

            t_idx_sdp_relx[n_sdp] = rows
            t_idx_sdp_subp[n_sdp] = map_rows_subp[rows]
            t = lhs_expr[rows]

            # Calculate dim where smat space dimensions are dim x dim
            dim = round(Int, sqrt(1/4+2*length(rows))-1/2)
            smat_sdp[n_sdp] = Symmetric(zeros(dim, dim))
            T_sdp[n_sdp] = T = Array{JuMP.AffExpr,2}(undef, dim, dim)

            # Set up smat arrays and set bounds
            k = 1
            for j in 1:dim, i in j:dim
                if j == i
                    @constraint(model_mip, t[k] >= 0)
                    T[i,j] = t[k]
                else
                    T[i,j] = T[j,i] = sqrt2inv*t[k]
                end
                k += 1
            end

            # Add initial SDP outer approximation cuts
            if m.init_sdp_lin
                # Using linear initial cuts: initial OA polyhedron is the dual cone of diagonally dominant matrices (an important subset of the PSD cone)
                for j in 1:dim, i in (j+1):dim
                    # K* cuts on (T_ii, T_jj, T_ij) are (1, 1, (+/-) 2)
                    @constraint(model_mip, T[i,i] + T[j,j] - 2*T[i,j] >= 0)
                    @constraint(model_mip, T[i,i] + T[j,j] + 2*T[i,j] >= 0)
                end
            elseif m.init_sdp_soc
                # Using SOC initial cuts: initial OA set (SOC-representable) is the dual cone of scaled diagonally dominant matrices (an important subset of the PSD cone; enforces 2x2 principal submatrix PSDness; implies the linear initial cuts)
                for j in 1:dim, i in (j+1):dim
                    # 3-dim rotated-SOC K* cut is (T_ii, T_jj, sqrt2*T_ij) in RSOC^3
                    # Use norm to add SOC constraint
                    # (p1, p2, q) in RSOC <-> (p1+p2, p1-p2, sqrt2*q) in SOC
                    @constraint(model_mip, T[i,i] + T[j,j] - norm(JuMP.AffExpr[(T[i,i] - T[j,j]), 2*T[i,j]]) >= 0)
                end
            end
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_int = x_all[cols_int]
    m.x_cont = x_all[cols_cont]
    # @show model_mip

    m.num_cones = n_lin + n_soc + n_exp + n_sdp

    if m.soc_in_mip
        m.num_soc = 0
    end
    m.r_idx_soc_subp = r_idx_soc_subp
    m.t_idx_soc_subp = t_idx_soc_subp
    m.r_soc = r_soc
    m.t_soc = t_soc
    m.pi_soc = pi_soc
    m.rho_soc = rho_soc

    m.r_idx_exp_subp = r_idx_exp_subp
    m.s_idx_exp_subp = s_idx_exp_subp
    m.t_idx_exp_subp = t_idx_exp_subp
    m.r_exp = r_exp
    m.s_exp = s_exp
    m.t_exp = t_exp

    m.t_idx_sdp_subp = t_idx_sdp_subp
    m.smat_sdp = smat_sdp
    m.T_sdp = T_sdp

    return (r_idx_soc_relx, t_idx_soc_relx, r_idx_exp_relx, s_idx_exp_relx, t_idx_exp_relx, t_idx_sdp_relx)
end


#=========================================================
 Iterative and MSD algorithm functions
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m)
    count_subopt = 0

    while true
        if (time() - m.logs[:total]) > m.timeout
            return :UserLimit
        end

        if count_subopt < m.mip_subopt_count
            # Solve is a partial solve: use subopt MIP solver, trust that user has provided reasonably small time limit
            setsolver(m.model_mip, m.mip_subopt_solver)
            count_subopt += 1
        else
            # Solve is a full solve: use full MIP solver with remaining time limit
            if isfinite(m.timeout) && applicable(MathProgBase.setparameters!, m.mip_solver)
                MathProgBase.setparameters!(m.mip_solver, TimeLimit=max(1., m.timeout - (time() - m.logs[:total])))
            end
            setsolver(m.model_mip, m.mip_solver)
            count_subopt = 0
        end

        if m.use_mip_starts && isfinite(m.best_obj)
            # Give the best feasible solution to the MIP as a warm-start
            m.logs[:n_add] += 1
            set_best_soln!(m, m.best_int, m.best_cont)
        else
            # For MIP solvers that accept warm starts without checking feasibility, set all variables to NaN
            for i in 1:MathProgBase.numvar(m.model_mip)
                setvalue(JuMP.Variable(m.model_mip, i), NaN)
            end
        end

        # Solve MIP
        start_time_mip = time()
        status_mip = solve(m.model_mip, suppress_warnings=true)
        m.logs[:mip_solve] += time() - start_time_mip
        m.logs[:n_iter] += 1

        # End if MIP didn't stop because of (sub)optimal or user limit
        if (status_mip != :UserLimit) && (status_mip != :Optimal) && (status_mip != :Suboptimal)
            return status_mip
        end

        # Update best bound from MIP bound
        mip_obj_bound = getobjbound(m.model_mip)
        if isfinite(mip_obj_bound) && (mip_obj_bound > m.best_bound)
            m.best_bound = mip_obj_bound
        end

        if !isfinite(getobjectivevalue(m.model_mip))
            if count_subopt > 0
                # Solve was not an optimal solve and MIP solver doesn't have a feasible solution, finish iteration and make next solve optimal
                count_subopt = m.mip_subopt_count
                @warn "MIP objective is NaN, proceeding to next optimal MIP solve\n"
                continue
            else
                # Hit user limit, must end
                return status_mip
            end
        end

        # Try to solve new conic subproblem and add subproblem cuts, update incumbent solution if feasible conic solution
        is_viol_subp = solve_subp_add_subp_cuts!(m, true)
        is_viol_any = is_viol_subp

        if m.sep_cuts_assist
            # Try to add primal cuts on MIP solution, update incumbent if feasible
            (is_feas, is_viol_any) = check_feas_add_sep_cuts!(m, ((m.sep_cuts_assist && !is_viol_subp) || m.sep_cuts_always))
        end

        # Update gap if best bound and best objective are finite
        if isfinite(m.best_obj) && isfinite(m.best_bound)
            m.gap_rel_opt = (m.best_obj - m.best_bound) / (abs(m.best_obj) + 1e-5)
        end

        # Print iteration information
        print_gap(m)

        if m.gap_rel_opt <= m.rel_gap
            # Opt gap condition satisfied
            return :Optimal
        elseif count_subopt > 0
            # MIP solve was suboptimal, try solving next MIP to optimality, if that doesn't help then we will end on next iteration
            count_subopt = m.mip_subopt_count
        elseif !is_viol_subp && !is_viol_any
            # MIP solve was optimal, no violated cuts were added, so must finish
            if status_mip == :UserLimit
                return :UserLimit
            elseif isfinite(m.gap_rel_opt)
                return :Suboptimal
            else
                return :FailedOA
            end
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
        m.logs[:n_lazy] += 1

        # Update best bound from MIP bound
        mip_obj_bound = MathProgBase.cbgetbestbound(cb)
        if isfinite(mip_obj_bound) && (mip_obj_bound > m.best_bound)
            m.best_bound = mip_obj_bound
        end

        # Try to solve new conic subproblem and add subproblem cuts, update incumbent solution if feasible conic solution
        is_viol_subp = solve_subp_add_subp_cuts!(m, true)

        # Try to add primal cuts on MIP solution, update incumbent if feasible
        if m.sep_cuts_assist
            (is_feas, is_viol_any) = check_feas_add_sep_cuts!(m, ((m.sep_cuts_assist && !is_viol_subp) || m.sep_cuts_always))

            # If solution is infeasible but we added no cuts, warn because MIP solver could accept a bad solution
            if !is_feas && !is_viol_subp && !is_viol_any
                @warn "Lazy callback solution is infeasible but no cuts could be added\n"
            end
        end

        # Update gap if best bound and best objective are finite
        if isfinite(m.best_obj) && isfinite(m.best_bound)
            m.gap_rel_opt = (m.best_obj - m.best_bound) / (abs(m.best_obj) + 1e-5)
            if m.gap_rel_opt < m.rel_gap
                # Gap condition satisfied, stop MIP solve
                return JuMP.StopTheSolver
            end
        end
    end
    addlazycallback(m.model_mip, callback_lazy)

    if m.use_mip_starts
        # Add heuristic callback to give MIP solver feasible solutions from conic solves
        function callback_heur(cb)
            # If have a new best feasible solution since last heuristic solution added, set MIP solution to the new best feasible solution
            m.logs[:n_heur] += 1
            if m.new_incumb
                m.logs[:n_add] += 1
                m.cb_heur = cb
                set_best_soln!(m, m.best_int, m.best_cont)
                addsolution(cb)
                m.new_incumb = false
            end
        end
        addheuristiccallback(m.model_mip, callback_heur)
    end

    # Start MIP solver
    m.logs[:mip_solve] = time()
    status_mip = solve(m.model_mip, suppress_warnings=true)
    m.logs[:mip_solve] = time() - m.logs[:mip_solve]

    # End if MIP didn't stop because of optimal or user limit
    if (status_mip != :UserLimit) && (status_mip != :Optimal) && (status_mip != :Suboptimal)
        return status_mip
    end

    # Update best bound from MIP bound
    mip_obj_bound = getobjbound(m.model_mip)
    if isfinite(mip_obj_bound) && (mip_obj_bound > m.best_bound)
        m.best_bound = mip_obj_bound
    end

    if isfinite(getobjectivevalue(m.model_mip))
        # Check final mip solver solution - if using sep cuts assist, accept if feasible and new
        solve_subp_add_subp_cuts!(m, false)
        if m.sep_cuts_assist
            check_feas_add_sep_cuts!(m, false)
        end
    end

    # Update gap if best bound and best objective are finite
    if isfinite(m.best_obj) && isfinite(m.best_bound)
        m.gap_rel_opt = (m.best_obj - m.best_bound) / (abs(m.best_obj) + 1e-5)
    end

    if m.gap_rel_opt <= m.rel_gap
        # Opt gap condition satisfied
        return :Optimal
    elseif status_mip == :UserLimit
        return :UserLimit
    elseif isfinite(m.gap_rel_opt)
        return :Suboptimal
    else
        return :FailedOA
    end
end


#=========================================================
 Miscellaneous algorithm functions
=========================================================#

# Construct and warm-start MIP solution using given solution
function set_best_soln!(m, soln_int, soln_cont)
    if m.mip_solver_drives && m.oa_started
        for j in 1:length(m.x_int)
            setsolutionvalue(m.cb_heur, m.x_int[j], soln_int[j])
        end
        for j in 1:length(m.x_cont)
            setsolutionvalue(m.cb_heur, m.x_cont[j], soln_cont[j])
        end
    else
        for j in 1:length(m.x_int)
            setvalue(m.x_int[j], soln_int[j])
        end
        for j in 1:length(m.x_cont)
            setvalue(m.x_cont[j], soln_cont[j])
        end
    end

    for n in 1:m.num_soc
        r_val = getvalue(m.r_soc[n])
        t_val = getvalue(m.t_soc[n])
        dim = length(t_val)

        if m.soc_disagg
            # Set disaggregated SOC variable pi values
            if r_val < m.cut_zero_tol
                # r value is small so set pi to 0
                pi_val = zeros(dim)
            else
                # r value is significantly positive so calculate pi values
                pi_val = (t_val.^2)/(2*r_val)
            end

            pi = m.pi_soc[n]
            if m.mip_solver_drives && m.oa_started
                for j in 1:dim
                    setsolutionvalue(m.cb_heur, pi[j], pi_val[j])
                end
            else
                for j in 1:dim
                    setvalue(pi[j], pi_val[j])
                end
            end
        end

        if m.soc_abslift
            # Set absval lifted variable rho values
            rho = m.rho_soc[n]
            rho_val = abs.(t_val)
            if m.mip_solver_drives && m.oa_started
                for j in 1:dim
                    setsolutionvalue(m.cb_heur, rho[j], rho_val[j])
                end
            else
                for j in 1:dim
                    setvalue(rho[j], rho_val[j])
                end
            end
        end
    end
end

# Transform svec vector into symmetric smat matrix
function make_smat!(smat::Symmetric{Float64,Array{Float64,2}}, svec::Vector{Float64})
    # smat is uplo U Symmetric
    dim = size(smat, 1)
    k = 1

    for i in 1:dim, j in i:dim
        if i == j
            smat.data[i,j] = svec[k]
        else
            smat.data[i,j] = sqrt2inv*svec[k]
        end
        k += 1
    end

    return smat
end

# Remove near-zeros from an array, return false if all values are near-zeros
function clean_array!(m, data::Array{Float64})
    any_large = false

    for j in 1:length(data)
        if abs(data[j]) < m.cut_zero_tol
            data[j] = 0.
        elseif !any_large
            any_large = true
        end
    end

    return any_large
end

# Check and record violation and add cut, return true if violated
function add_cut!(m, cut_expr, cone_logs)
    is_viol_cut = false

    if !m.all_disagg
        # Not disaggregating the nonpolyhedral cone cuts, so build up single cut and add it after iterate over the cones
        m.aggregate_cut += cut_expr
    elseif !m.oa_started
        # Add non-lazy cut to OA model
        @constraint(m.model_mip, cut_expr >= 0)
        cone_logs[:n_relax] += 1
    else
        cut_val = getvalue(cut_expr)
        if !m.viol_cuts_only || (cut_val <= -m.mip_feas_tol)
            if m.mip_solver_drives
                # Add lazy cut
                @lazyconstraint(m.cb_lazy, cut_expr >= 0)
            else
                # Add non-lazy cut
                @constraint(m.model_mip, cut_expr >= 0)
            end

            # Check if cut is violated by current solution
            if cut_val <= -m.mip_feas_tol
                is_viol_cut = true
                cone_logs[:n_viol_total] += 1
            else
                cone_logs[:n_nonviol_total] += 1
            end
        end
    end

    return is_viol_cut
end


#=========================================================
 Subproblem functions
=========================================================#

# Solve the subproblem for the current integer solution, add new incumbent conic solution if feasible and best, add K* cuts from subproblem dual solution
function solve_subp_add_subp_cuts!(m, add_cuts::Bool)
    # Get current integer solution
    soln_int = getvalue(m.x_int)
    if m.round_mip_sols
        # Round the integer values
        soln_int = map!(round, soln_int)
    end

    if haskey(m.cache_dual, soln_int)
        # Integer solution has been seen before, cannot get new subproblem cuts
        m.logs[:n_repeat] += 1

        if !add_cuts || !m.mip_solver_drives || m.sep_cuts_only || !m.solve_subp
            if !m.mip_solver_drives && !m.sep_cuts_only
                @warn "Repeated integer solution without converging\n"
            end
            # Nothing to do if using iterative or if not using subproblem cuts
            return false
        else
            # In MSD, re-add subproblem cuts from existing conic dual
            dual_conic = m.cache_dual[soln_int]
            if isempty(dual_conic)
                # Don't have a conic dual due to conic failure, nothing to do
                return false
            end
        end
    elseif !m.solve_subp
        return false
    else
        # Integer solution is new, save it
        m.cache_dual[soln_int] = Float64[]

        # Calculate new b vector from integer solution and solve conic subproblem model
        b_sub_int = m.b_sub - m.A_sub_int*soln_int
        (status_conic, soln_conic, dual_conic) = solve_subp!(m, b_sub_int)

        # Handle a primal solution
        if !isempty(soln_conic)
            # Calculate full objective value and check if incumbent
            obj_full = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_conic)
            if obj_full < m.best_obj
                # Conic solver solution is a new incumbent
                m.best_obj = obj_full
                m.best_int = soln_int
                m.best_cont = soln_conic
                m.new_incumb = true
                m.is_best_conic = true
            end
            m.logs[:n_feas_conic] += 1
        end

        # If not using subproblem cuts, return
        if m.sep_cuts_only || !add_cuts
            return false
        end

        # Handle a dual solution/ray
        if !isempty(dual_conic)
            if status_conic == :Infeasible
                # Calculate obj value of dual ray and check it is strictly positive
                dual_value = -dot(b_sub_int, dual_conic)
                if dual_value < 1e-10
                    @warn "For infeasible subproblem, dual ray objective value $dual_value is not significantly positive (please submit an issue)\n"
                elseif m.scale_subp_cuts
                    # Rescale using dual value for dual infeasibility case
                    rmul!(dual_conic, m.inf_subp_scale/dual_value)
                end
            else
                # Calculate obj value of full dual solution
                dual_value = -dot(b_sub_int, dual_conic) + dot(m.c_sub_int, soln_int)
                if m.scale_subp_cuts
                    # Rescale using dual value for strong duality case
                    rmul!(dual_conic, m.opt_subp_scale/(abs(dual_value) + 1e-5))
                end
            end

            # In MSD, save the dual so can re-add subproblem cuts later
            if m.mip_solver_drives
                m.cache_dual[soln_int] = dual_conic
            end
        else
            return false
        end
    end

    # Add K* cut(s) from subproblem dual solution/ray
    start_time_subp_cuts = time()
    is_viol_any = false
    if !m.all_disagg
        m.aggregate_cut = JuMP.AffExpr(0)
    end

    for n in 1:m.num_soc
        u_val = dual_conic[m.r_idx_soc_subp[n]]
        w_val = dual_conic[m.t_idx_soc_subp[n]]
        is_viol_any |= add_subp_cut_soc!(m, m.r_soc[n], m.t_soc[n], m.pi_soc[n], m.rho_soc[n], u_val, w_val)
    end

    for n in 1:m.num_exp
        u_val = dual_conic[m.r_idx_exp_subp[n]]
        v_val = dual_conic[m.s_idx_exp_subp[n]]
        w_val = dual_conic[m.t_idx_exp_subp[n]]
        is_viol_any |= add_subp_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], u_val, v_val, w_val)
    end

    for n in 1:m.num_sdp
        # Get smat space dual
        W_val = make_smat!(m.smat_sdp[n], dual_conic[m.t_idx_sdp_subp[n]])
        is_viol_any |= add_subp_cut_sdp!(m, m.T_sdp[n], W_val)
    end

    if !m.all_disagg
        @assert !is_viol_any
        cut_val = getvalue(m.aggregate_cut)

        if !m.viol_cuts_only || (cut_val <= -m.mip_feas_tol)
            if m.mip_solver_drives
                # Add lazy cut
                @lazyconstraint(m.cb_lazy, m.aggregate_cut >= 0)
            else
                # Add non-lazy cut
                @constraint(m.model_mip, m.aggregate_cut >= 0)
            end

            # Check if cut is violated by current solution
            if cut_val <= -m.mip_feas_tol
                is_viol_any = true
            end
        end
    end

    m.logs[:subp_cuts] += time() - start_time_subp_cuts

    return is_viol_any
end

# Solve conic subproblem given some solution to the integer variables, update incumbent
function solve_subp!(m, b_sub_int::Vector{Float64})
    # Load/solve conic model
    start_time_subp_solve = time()
    if m.update_conicsub
        # Reuse model already created by changing b vector
        MathProgBase.setbvec!(m.model_conic, b_sub_int)
    else
        # Load all data
        MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, b_sub_int, m.cone_con_sub, m.cone_var_sub)
    end

    m.logs[:n_conic] += 1

    # Optionally dump the conic subproblem into a cbf file
    if m.dump_subproblems
        dat = ConicBenchmarkUtilities.mpbtocbf(string(m.logs[:n_conic]), m.c_sub_cont, m.A_sub_cont, b_sub_int, m.cone_con_sub, m.cone_var_sub, fill(:Cont, length(m.c_sub_cont)))
        ConicBenchmarkUtilities.writecbfdata((m.dump_basename * "_" * string(time()) * ".cbf"), dat)
    end

    MathProgBase.optimize!(m.model_conic)
    m.logs[:subp_solve] += time() - start_time_subp_solve

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
        @warn "Conic solver failure: returned status $status_conic\n"
        m.logs[:n_fail] += 1
    else
        @warn "Conic solver failure: returned status $status_conic\n"
        m.logs[:n_other] += 1
    end

    # Get a dual
    # if status_conic in (:Optimal, :Infeasible, :Suboptimal, :PDFeas, :DualFeas)
    dual_conic = Float64[]
    try
        dual_conic = MathProgBase.getdual(m.model_conic)
        if any(isnan, dual_conic)
            dual_conic = Float64[]
        end
    catch
    end

    # Get a primal
    soln_conic = Float64[]
    if status_conic in (:Optimal, :Suboptimal, :PDFeas, :PrimFeas)
        soln_conic = MathProgBase.getsolution(m.model_conic)
        if any(isnan, soln_conic)
            soln_conic = Float64[]
        end
    end

    # Free the conic model if not saving it
    if !m.update_conicsub && applicable(MathProgBase.freemodel!, m.model_conic)
        MathProgBase.freemodel!(m.model_conic)
    end

    return (status_conic, soln_conic, dual_conic)
end


#=========================================================
 Subproblem K* cut functions
=========================================================#

# Add a SOC subproblem cut: (u,w) in SOC* = SOC <-> u >= norm2(w) >= 0
function add_subp_cut_soc!(m, r, t, pi, rho, u_val, w_val)
    is_viol_cut = false

    if clean_array!(m, w_val)
        # K* projected subproblem cut is (norm2(w), w)
        u_val = norm(w_val)

        if m.scale_subp_up && (u_val < 1.)
            # Scale up to equivalent separation cut, with u = 1
            w_val /= u_val
            u_val = 1.
        end

        is_viol_cut = add_cut_soc!(m, r, t, pi, rho, u_val, w_val)
        m.logs[:SOC][:n_subp] += 1
    end

    return is_viol_cut
end

# Add an Exp subproblem cut: (w,v,u) in ExpDual <-> (w < 0 && u >= -w*exp(v/w - 1) > 0) || (w = 0 && u >= 0 && v >= 0)
function add_subp_cut_exp!(m, r, s, t, u_val, v_val, w_val)
    is_viol_cut = false

    if w_val > -1e-8
        # w is (near) zero: K* projected subproblem cut on (r,s,t) is (max(u, 0), max(v, 0), 0)
        if (u_val > m.cut_zero_tol) && (v_val > m.cut_zero_tol)

            if m.scale_subp_up && (u_val < 1.) && (v_val < 1.)
                # Scale up so largest is 1
                uvmax = max(u_val, v_val)
                u_val /= uvmax
                v_val /= uvmax
            end

            is_viol_cut = add_cut_exp!(m, r, s, t, u_val, v_val, 0.)
            m.logs[:ExpPrimal][:n_subp] += 1
        end
    else
        # w is significantly negative: K* projected subproblem cut on (r,s,t) is (-w*exp(v/w - 1), v, w)
        u_val = -w_val*exp(v_val/w_val - 1)

        if m.scale_subp_up && (u_val < 1.)
            # Scale up to equivalent separation cut, with u = 1
            v_val /= u_val
            w_val /= u_val
            u_val = 1.
        end

        is_viol_cut = add_cut_exp!(m, r, s, t, u_val, v_val, w_val)
        m.logs[:ExpPrimal][:n_subp] += 1
    end

    return is_viol_cut
end

# Add a PSD subproblem cut: W in SDP* = SDP <-> sum_{j: lambda_j < 0} lambda_j(W) = 0
function add_subp_cut_sdp!(m, T, W_val)
    is_viol_cut = false

    W_eig_obj = eigen!(W_val, m.cut_zero_tol, Inf)

    # K* projected (scaled) subproblem cut is sum_{j: lambda_j > 0} lambda_j W_eig_j W_eig_j'
    if !isempty(W_eig_obj.values)
        sqrteig = sqrt.(W_eig_obj.values)

        if m.scale_subp_up
            # Scale up to equivalent separation cuts, with lambdas >= 1
            for j in 1:length(sqrteig)
                if sqrteig[j] < 1.
                    sqrteig[j] = 1.
                end
            end
        end

        W_eig = W_eig_obj.vectors*Diagonal(sqrteig)
        if clean_array!(m, W_eig)
            is_viol_cut = add_cut_sdp!(m, T, W_eig)
            m.logs[:SDP][:n_subp] += 1
        end
    end

    return is_viol_cut
end


#=========================================================
 Conic feasibility and separation K* cut functions
=========================================================#

# Check cone infeasibilities of current solution, add K* cuts from current solution for infeasible cones, if feasible check new incumbent
function check_feas_add_sep_cuts!(m, add_cuts::Bool)
    start_time_sep_cuts = time()
    is_viol_any = false
    max_viol = 0.

    for n in 1:m.num_soc
        (is_viol_cut, viol) = add_sep_cut_soc!(m, add_cuts, m.r_soc[n], m.t_soc[n], m.pi_soc[n], m.rho_soc[n])
        is_viol_any |= is_viol_cut
        max_viol = max(viol, max_viol)
    end

    for n in 1:m.num_exp
        (is_viol_cut, viol) = add_sep_cut_exp!(m, add_cuts, m.r_exp[n], m.s_exp[n], m.t_exp[n])
        is_viol_any |= is_viol_cut
        max_viol = max(viol, max_viol)
    end

    for n in 1:m.num_sdp
        (is_viol_cut, viol) = add_sep_cut_sdp!(m, add_cuts, m.T_sdp[n])
        is_viol_any |= is_viol_cut
        max_viol = max(viol, max_viol)
    end

    m.logs[:sep_cuts] += time() - start_time_sep_cuts

    # Check feasibility of solution (via worst cone violation) and return whether feasible and whether added violated cut
    if max_viol < feas_factor*m.mip_feas_tol
        # Accept MIP solution as feasible and check if new incumbent
        m.logs[:n_feas_mip] += 1
        soln_int = getvalue(m.x_int)
        soln_cont = getvalue(m.x_cont)
        obj_full = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_cont)

        if obj_full < m.best_obj
            # Save new incumbent info
            m.best_obj = obj_full
            m.best_int = soln_int
            m.best_cont = soln_cont
            m.is_best_conic = false
        end

        return (true, is_viol_any)
    else
        return (false, is_viol_any)
    end
end

# Calculate the SOC violation and optionally add a separation cut
function add_sep_cut_soc!(m, add_cuts::Bool, r, t, pi, rho)
    is_viol_cut = false
    viol = 0.

    r_val = getvalue(r)
    t_val = getvalue(t)

    # Violation is norm(t) - r
    viol = norm(t_val) - r_val

    # K* separation cut is (1, -t/norm(t))
    if add_cuts && (viol > m.mip_feas_tol) && clean_array!(m, t_val)
        w_val = -t_val/norm(t_val)
        is_viol_cut = add_cut_soc!(m, r, t, pi, rho, 1., w_val)
        m.logs[:SOC][:n_sep] += 1
    end

    return (is_viol_cut, viol)
end

# Calculate the Exp violation and optionally add a separation cut
function add_sep_cut_exp!(m, add_cuts::Bool, r, s, t)
    is_viol_cut = false
    viol = 0.

    r_val = getvalue(r)
    s_val = getvalue(s)
    t_val = getvalue(t)

    if s_val <= 1e-7
        # s is (almost) zero: violation is t
        viol = t_val

        if add_cuts && (viol > m.mip_feas_tol)
            # TODO: for now, error if r is too small
            if r_val <= 1e-12
                error("Cannot add exp cone separation cut on point ($r_val, $s_val, $t_val)\n")
            end

            # K* separation cut on (r,s,t) is (t/r, -2*log(exp(1)*t/2r), -2)
            u_val = t_val/r_val
            v_val = -2.0 * (1.0 + log(u_val/2.0))
            is_viol_cut = add_cut_exp!(m, r, s, t, u_val, v_val, -2.)
            m.logs[:ExpPrimal][:n_sep] += 1
        end
    else
        # s is significantly positive: violation is s*exp(t/s) - r
        ets = exp(t_val/s_val)
        viol = s_val*ets - r_val

        if add_cuts && (viol > m.mip_feas_tol)
            # K* separation cut on (r,s,t) is (1, (t-s)/s*exp(t/s), -exp(t/s))
            v_val = (t_val - s_val)/s_val*ets
            is_viol_cut = add_cut_exp!(m, r, s, t, 1., v_val, -ets)
            m.logs[:ExpPrimal][:n_sep] += 1
        end
    end

    return (is_viol_cut, viol)
end

# Calculate the PSD violation and optionally add a separation cut
function add_sep_cut_sdp!(m, add_cuts::Bool, T)
    is_viol_cut = false
    viol = 0.

    # Get eigendecomposition object, with eigenvalues smaller than separation cut feasibility tolerance
    T_eig_obj = eigen!(Symmetric(getvalue(T)), -Inf, -m.mip_feas_tol)

    # Violation is negative min eigenvalue (empty if all eigenvalues larger than separation cut feasibility tolerance)
    if !isempty(T_eig_obj.values)
        viol = -minimum(T_eig_obj.values)

        # K* separation cut is sum_{j: lambda_j < 0} T_eig_j T_eig_j'
        T_eig = T_eig_obj.vectors
        if add_cuts && clean_array!(m, T_eig)
            is_viol_cut = add_cut_sdp!(m, T, T_eig)
            m.logs[:SDP][:n_sep] += 1
        end
    end

    return (is_viol_cut, viol)
end


#=========================================================
 Specific cone K* cut functions
=========================================================#

# Add K* cuts for SOC: (u,w) in SOC* = SOC <-> u >= norm2(w) >= 0
function add_cut_soc!(m, r, t, pi, rho, u_val, w_val)
    is_viol_cut = false

    dim = length(w_val)
    add_full = false

    if m.soc_disagg
        for j in 1:dim
            if w_val[j] == 0.
                continue
            elseif abs(w_val[j]) < u_val*unstable_soc_disagg_tol
                # Cut is poorly conditioned so add full SOC cut later
                add_full = true
                m.logs[:n_unst_soc] += 1
            end

            if m.soc_abslift
                # Disaggregated K* cut on (r, pi_j, rho_j) is ((w_j/u)^2/2, 1, -|w_j/u|)
                # Scale by dim*u_val
                cut_expr = dim*w_val[j]^2/(2.0 * u_val)*r + dim*u_val*pi[j] - dim*abs(w_val[j])*rho[j]
            else
                # Disaggregated K* cut on (r, pi_j, t_j) is ((w_j/u)^2/2, 1, w_j/u)
                # Scale by dim*u_val
                cut_expr = dim*w_val[j]^2/(2.0 * u_val)*r + dim*u_val*pi[j] + dim*w_val[j]*t[j]
            end

            is_viol_cut |= add_cut!(m, cut_expr, m.logs[:SOC])
        end
    end

    if add_full || !m.soc_disagg
        if m.soc_abslift
            # Non-disaggregated K* cut on (r, rho_1, ..., rho_j, ..., rho_dim) is (u, -|w_j|)
            cut_expr = u_val*r - sum(abs(w_val[j])*rho[j] for j in 1:dim)
        else
            # Non-disaggregated K* cut on (r, t) is (u, w)
            cut_expr = u_val*r + dot(w_val, t)
        end

        is_viol_cut |= add_cut!(m, cut_expr, m.logs[:SOC])
    end

    return is_viol_cut
end

# Add K* cut for Exp: (w,v,u) in ExpDual <-> (w < 0 && u >= -w*exp(v/w - 1) > 0) || (w = 0 && u >= 0 && v >= 0)
function add_cut_exp!(m, r, s, t, u_val, v_val, w_val)
    cut_expr = u_val*r + v_val*s + w_val*t

    return add_cut!(m, cut_expr, m.logs[:ExpPrimal])
end

# Add K* cuts for PSD: W in SDP* = SDP <-> sum_{j: lambda_j < 0} lambda_j(W) = 0
function add_cut_sdp!(m, T, W_eig)
    is_viol_cut = false

    (dim, num_eig) = size(W_eig)

    if m.sdp_eig
        # Using PSD eigenvector cuts
        for j in 1:num_eig
            W_eig_j = W_eig[:,j]

            if m.sdp_soc && !(m.mip_solver_drives && m.oa_started)
                # Using SDP SOC eig cuts (cannot add SOC cuts during callbacks)
                # Over all diagonal entries i, exclude the largest one
                i = findmax(abs.(W_eig_j))[2]

                # 3-dim rotated-SOC K* constraint is (T_i,i, <T_-i,-i, (W_eig_j*W_eig_j')_-i,-i>, sqrt2*<T_-i,i, W_eig_j_-i>) in RSOC^3
                # Use norm to add SOC constraint
                # (p1, p2, q) in RSOC <-> (p1+p2, p1-p2, sqrt2*q) in SOC
                p2 = sum(T[k,l]*W_eig_j[k]*W_eig_j[l] for k in 1:dim, l in 1:dim if (k!=i && l!=i))
                cut_expr = T[i,i] + p2 - norm([(T[i,i] - p2), 2.0 * sum(T[k,i]*W_eig_j[k] for k in 1:dim if k!=i)])
            else
                # Using SDP linear eig cuts
                # K* cut on T is W_eig_j*W_eig_j'
                # Scale by num_eig
                cut_expr = num_eig*dot(Symmetric(W_eig_j*W_eig_j'), T)
            end

            is_viol_cut |= add_cut!(m, cut_expr, m.logs[:SDP])
        end
    else
        # Using full PSD cut
        W = Symmetric(W_eig*W_eig')

        if m.sdp_soc && !(m.mip_solver_drives && m.oa_started)
            # Using SDP SOC full cut (cannot add SOC cuts during callbacks)
            # Over all diagonal entries i, exclude the largest one
            i = findmax(abs.(diag(W)))[2]

            # (num_eig+2)-dim rotated-SOC K* constraint is (T_i,i, <T_-i,-i, W_-i,-i>, sqrt2*<T_-i,i, W_eig_1_-i>, ..., sqrt2*<T_-i,i, W_eig_num_eig_-i>) in RSOC^{num_eig+2}, where num_eig is the number of eigenvectors
            # Use norm to add SOC constraint
            # (p1, p2, q) in RSOC <-> (p1+p2, p1-p2, sqrt2*q) in SOC
            p2 = sum(T[k,l]*W[k,l] for k in 1:dim, l in 1:dim if (k!=i && l!=i))
            cut_expr = T[i,i] + p2 - norm([(T[i,i] - p2), 2.0 * [sum((T[k,i]*W_eig[k,j]) for k in 1:dim if k!=i) for j in 1:num_eig]...])
        else
            # Using SDP linear full cut
            # K* cut on T is W
            cut_expr = dot(W, T)
        end

        is_viol_cut |= add_cut!(m, cut_expr, m.logs[:SDP])
    end

    return is_viol_cut
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
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:subp_solve] = 0. # Solving conic subproblem model
    logs[:relax_cuts] = 0.  # Deriving and adding conic relaxation cuts
    logs[:subp_cuts] = 0.   # Deriving and adding subproblem cuts
    logs[:sep_cuts] = 0.   # Deriving and adding primal cuts

    # Counters
    logs[:n_lazy] = 0       # Number of times lazy is called in MSD
    logs[:n_iter] = 0       # Number of iterations in iterative
    logs[:n_repeat] = 0     # Number of times integer solution repeats
    logs[:n_conic] = 0      # Number of unique integer solutions (conic subproblem solves)
    logs[:n_inf] = 0        # Number of conic subproblem infeasible statuses
    logs[:n_opt] = 0        # Number of conic subproblem optimal statuses
    logs[:n_sub] = 0        # Number of conic subproblem suboptimal statuses
    logs[:n_lim] = 0        # Number of conic subproblem user limit statuses
    logs[:n_fail] = 0       # Number of conic subproblem conic failure statuses
    logs[:n_other] = 0      # Number of conic subproblem other statuses
    logs[:n_feas_conic] = 0 # Number of times get a new feasible solution from conic solver
    logs[:n_feas_mip] = 0   # Number of times get a new feasible solution from MIP solver
    logs[:n_heur] = 0       # Number of times heuristic is called in MSD
    logs[:n_add] = 0        # Number of times add new solution to MIP solver
    logs[:n_unst_soc] = 0   # Number of numerically unstable disaggregated SOC cuts

    # Cuts counters
    for cone in (:SOC, :ExpPrimal, :SDP)
        logs[cone] = Dict{Symbol,Any}()
        logs[cone][:n_sep] = 0
        logs[cone][:n_subp] = 0
        logs[cone][:n_relax] = 0
        logs[cone][:n_viol_total] = 0
        logs[cone][:n_nonviol_total] = 0
    end

    m.logs = logs
end

# Print objective gap information for iterative
function print_gap(m)
    if m.log_level >= 1
        if (m.logs[:n_iter] == 1) || (m.log_level > 2)
            @printf "\n%-5s | %-14s | %-14s | %-11s | %-11s\n" "Iter." "Best feasible" "Best bound" "Rel. gap" "Time (s)"
        end
        if m.gap_rel_opt < 1000
            @printf "%5d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" m.logs[:n_iter] m.best_obj m.best_bound m.gap_rel_opt (time() - m.logs[:total])
        else
            @printf "%5d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.logs[:n_iter] m.best_obj m.best_bound (isnan(m.gap_rel_opt) ? "Inf" : ">1000") (time() - m.logs[:total])
        end
        flush(stdout)
        flush(stderr)
    end
end

# Print after finish
function print_finish(m::PajaritoConicModel)
    ll = m.log_level

    if m.gap_rel_opt < -10*m.rel_gap
        # Warn if the best "feasible" solution has value better than the best OA bound (possible the conic solver solutions are not feasible for the MIP solver's tolerances)
        @warn "Solution value ($(m.best_obj)) is smaller than best bound ($(m.best_bound)): check solution feasibility (tightening primal feasibility tolerance of conic solver may help)\n"
        # m.status = :Error
        if ll > 0
            # Print more
            println("Pajarito will print diagnostic information")
            ll = 3
        end
    end

    if ll <= 0
        # Nothing to print
        return
    elseif !in(m.status, [:Optimal, :Suboptimal, :UserLimit, :Unbounded, :Infeasible])
        # Print more on a problematic status
        ll = 3
    end

    if m.mip_solver_drives
        @printf "\nMIP-solver-driven algorithm summary:\n"
    else
        @printf "\nIterative algorithm summary:\n"
    end
    @printf " - Status               = %14s\n" m.status
    @printf " - Best feasible        = %+14.6e\n" m.best_obj
    @printf " - Best bound           = %+14.6e\n" m.best_bound
    if m.gap_rel_opt < -10*m.rel_gap
        @printf " - Relative opt. gap    =*%14.3e*\n" m.gap_rel_opt
    else
        @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt
    end
    @printf " - Total time (s)       = %14.2e\n" m.logs[:total]

    if ll >= 3
        @printf "\nTimers (s):\n"
        @printf " - Setup                = %10.2e\n" (m.logs[:data_trans] + m.logs[:data_conic] + m.logs[:data_mip])
        @printf " -- Transform data      = %10.2e\n" m.logs[:data_trans]
        @printf " -- Create conic data   = %10.2e\n" m.logs[:data_conic]
        @printf " -- Create MIP data     = %10.2e\n" m.logs[:data_mip]
        @printf " - Algorithm            = %10.2e\n" (m.logs[:total] - (m.logs[:data_trans] + m.logs[:data_conic] + m.logs[:data_mip]))
        @printf " -- Solve relaxation    = %10.2e\n" m.logs[:relax_solve]
        @printf " -- Get relaxation cuts = %10.2e\n" m.logs[:relax_cuts]
        if m.mip_solver_drives
            @printf " -- MIP solver driving  = %10.2e\n" m.logs[:mip_solve]
        else
            @printf " -- Solve MIP models    = %10.2e\n" m.logs[:mip_solve]
        end
        @printf " -- Solve subproblems   = %10.2e\n" m.logs[:subp_solve]
        @printf " -- Get subproblem cuts = %10.2e\n" m.logs[:subp_cuts]
        @printf " -- Get separation cuts = %10.2e\n" m.logs[:sep_cuts]

        @printf "\nCounters:\n"
        if m.mip_solver_drives
            @printf " - Lazy callbacks       = %5d\n" m.logs[:n_lazy]
        else
            @printf " - Iterations           = %5d\n" m.logs[:n_iter]
        end
        @printf " -- Integer repeats     = %5d\n" m.logs[:n_repeat]
        @printf " -- Conic subproblems   = %5d\n" m.logs[:n_conic]
        if m.solve_subp
            @printf " --- Infeasible         = %5d\n" m.logs[:n_inf]
            @printf " --- Optimal            = %5d\n" m.logs[:n_opt]
            @printf " --- Suboptimal         = %5d\n" m.logs[:n_sub]
            @printf " --- UserLimit          = %5d\n" m.logs[:n_lim]
            @printf " --- ConicFailure       = %5d\n" m.logs[:n_fail]
            @printf " --- Other status       = %5d\n" m.logs[:n_other]
        end
        @printf " -- Feasible solutions  = %5d\n" (m.logs[:n_feas_conic] + m.logs[:n_feas_mip])
        @printf " --- From subproblems   = %5d\n" m.logs[:n_feas_conic]
        if !m.mip_solver_drives
            @printf " --- From MIP solve     = %5d\n" m.logs[:n_feas_mip]
        else
            @printf " --- In lazy callback   = %5d\n" m.logs[:n_feas_mip]
            @printf " - Heuristic callbacks  = %5d\n" m.logs[:n_heur]
            @printf " -- Solutions passed    = %5d\n" m.logs[:n_add]
        end

        @printf "\nSolution returned by %s solver\n" (m.is_best_conic ? "conic" : "MIP")
    end

    if ll >= 2
        if m.all_disagg
            @printf "\nRounds of full separation/subproblem cuts, and count of cuts added:"
            @printf "\n%-16s | %-6s | %-6s | %-6s | %-6s | %-6s\n" "Cone" "Subp." "Sep." "Total" "Relax." "Viol."
            for (cone, name) in zip((:SOC, :ExpPrimal, :SDP), ("Second order", "Primal expon.", "Pos. semidef."))
                log = m.logs[cone]
                if (log[:n_relax] + log[:n_viol_total] + log[:n_nonviol_total]) > 0
                    @printf "%16s | %6d | %6d | %6d | %6d | %6d\n" name log[:n_subp] log[:n_sep] (log[:n_viol_total] + log[:n_nonviol_total]) log[:n_relax] log[:n_viol_total]
                end
            end

            if m.num_soc > 0
                @printf "\n%d numerically unstable disaggregated SOC cuts\n" m.logs[:n_unst_soc]
            end
        end

        if isfinite(m.best_obj) && !any(isnan, m.final_soln)
            var_inf = calc_infeas(m.cone_var_orig, m.final_soln)
            con_inf = calc_infeas(m.cone_con_orig, m.b_orig-m.A_orig*m.final_soln)

            @printf "\nDistance to feasibility (negative indicates strict feasibility):"
            @printf "\n%-16s | %-9s | %-10s\n" "Cone" "Variable" "Constraint"
            for (v, c, name) in zip(var_inf, con_inf, ("Linear", "Second order", "Rotated S.O.", "Primal expon.", "Pos. semidef."))
                if isfinite(v) && isfinite(c)
                    @printf "%16s | %9.2e | %9.2e\n" name v c
                elseif isfinite(v)
                    @printf "%16s | %9.2e | %9s\n" name v "NA"
                elseif isfinite(c)
                    @printf "%16s | %9s | %9.2e\n" name "NA" c
                end
            end

            viol_int = -Inf
            viol_bin = -Inf
            for (j, vartype) in enumerate(m.var_types)
                if vartype == :Int
                    viol_int = max(viol_int, abs(m.final_soln[j] - round(m.final_soln[j])))
                elseif vartype == :Bin
                    if m.final_soln[j] < 0.5
                        viol_bin = max(viol_bin, abs(m.final_soln[j]))
                    else
                        viol_bin = max(viol_bin, abs(m.final_soln[j] - 1.))
                    end
                end
            end

            @printf "\nDistance to integrality of integer/binary variables:\n"
            if isfinite(viol_int)
                @printf "%16s | %9.2e\n" "integer" viol_int
            end
            if isfinite(viol_bin)
                @printf "%16s | %9.2e\n" "binary" viol_bin
            end
        end
    end

    println()
end

# Calculate absolute linear infeasibilities on each cone, and print worst
function calc_infeas(cones, vals)
    viol_lin = -Inf
    viol_soc = -Inf
    viol_rot = -Inf
    viol_exp = -Inf
    viol_sdp = -Inf

    for (cone, idx) in cones
        if cone == :Free
            nothing
        elseif cone == :Zero
            viol_lin = max(viol_lin, maximum(abs, vals[idx]))
        elseif cone == :NonNeg
            viol_lin = max(viol_lin, -minimum(vals[idx]))
        elseif cone == :NonPos
            viol_lin = max(viol_lin, maximum(vals[idx]))
        elseif cone == :SOC
            viol_soc = max(viol_soc, norm(vals[idx[j]] for j in 2:length(idx)) - vals[idx[1]])
        elseif cone == :SOCRotated
            # Convert to SOC and calculate using SOC violation function, maintain original scaling
            # (p1, p2, q) in RSOC <-> (sqrt2inv*(p1+p2), sqrt2inv*(-p1+p2), q) in SOC
            t = sqrt2inv*(vals[idx[1]] + vals[idx[2]])
            usqr = 1/2*(-vals[idx[1]] + vals[idx[2]])^2 + sum(abs2, vals[idx[j]] for j in 3:length(idx))
            viol_rot = max(viol_rot, sqrt(usqr) - t)
        elseif cone == :ExpPrimal
            if vals[idx[2]] <= 1e-7
                # s is (almost) zero: violation is t
                viol_exp = max(viol_exp, vals[idx[1]])
            else
                # s is significantly positive: violation is s*exp(t/s) - r
                viol_exp = max(viol_exp, vals[idx[2]]*exp(vals[idx[1]]/vals[idx[2]]) - vals[idx[3]])
            end
        elseif cone == :SDP
            dim = round(Int, sqrt(1/4+2*length(idx))-1/2)
            vals_smat = make_smat!(Symmetric(zeros(dim, dim)), vals[idx])
            viol_sdp = max(viol_sdp, -eigmin(vals_smat))
        end
    end

    return (viol_lin, viol_soc, viol_rot, viol_exp, viol_sdp)
end
