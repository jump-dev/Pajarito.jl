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


TODO issues
- MPB issue - can't call supportedcones on defaultConicsolver
- maybe want two zero tols: one for discarding cut if largest value is too small, and one for setting near zeros to zero (the former should be larger)
- log primal cuts added

TODO features
- implement warm-start pajarito: use set_best_soln!
- query logs information etc
- print cone info to one file and gap info to another file

TODO SDP
- use the outer infeasibility to choose which SDP cuts to add. maybe can pick the best dual cuts by using the current primal solution
- fix soc in mip: if using SOC in mip, can only detect slacks with coef -1 (or sqrt(2)?)

=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
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

    # Initial conic data
    num_var_orig::Int           # Initial number of variables
    num_con_orig::Int           # Initial number of constraints
    c_orig::Vector{Float64}     # Initial objective coefficients vector
    A_orig::SparseMatrixCSC{Float64,Int64} # Initial affine constraint matrix (sparse representation)
    b_orig::Vector{Float64}     # Initial constraint right hand side
    cone_con_orig::Vector{Tuple{Symbol,Vector{Int}}} # Initial constraint cones vector (cone, index)
    cone_var_orig::Vector{Tuple{Symbol,Vector{Int}}} # Initial variable cones vector (cone, index)
    var_types::Vector{Symbol}   # Variable types vector on original variables (only :Bin, :Cont, :Int)
    # var_start::Vector{Float64}  # Variable warm start vector on original variables

    # Transformed data
    cone_con_new::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data after converting variable cones to constraint cones
    cone_var_new::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data after converting variable cones to constraint cones and adding slacks
    num_con_new::Int            # Number of constraints after converting variable cones to constraint cones
    num_var_new::Int            # Number of variables after adding slacks
    b_new::Vector{Float64}      # Subvector of b containing full rows
    c_new::Vector{Float64}      # Objective coefficient subvector for continuous variables
    A_new::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    keep_cols::Vector{Int}      # Indices of variables in original problem that remain after removing zeros
    row_to_slckj::Dict{Int,Int} # Dictionary from row index in MIP to slack column index if slack exists
    row_to_slckv::Dict{Int,Float64} # Dictionary from row index in MIP to slack coefficient if slack exists

    # Conic constructed data
    cone_con_sub::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data in conic subproblem
    cone_var_sub::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data in conic subproblem
    map_rows_sub::Vector{Int} # Row indices in conic subproblem for nonlinear cone
    cols_cont::Vector{Int}      # Column indices of continuous variables in MIP
    cols_int::Vector{Int}       # Column indices of integer variables in MIP
    A_sub_cont::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    A_sub_int::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and integer variable columns
    b_sub::Vector{Float64}      # Subvector of b containing full rows
    c_sub_cont::Vector{Float64} # Subvector of c for continuous variables
    c_sub_int::Vector{Float64}  # Subvector of c for integer variables
    b_sub_int::Vector{Float64}  # Slack vector that we operate on in conic subproblem

    # MIP constructed data
    model_mip::JuMP.Model       # JuMP MIP (outer approximation) model
    x_mip::Vector{JuMP.Variable} # JuMP vector of original variables

    # SOC data
    num_soc::Int                # Number of SOCs
    summ_soc::Dict{Symbol,Real} # Data and infeasibilities
    dim_soc::Vector{Int}        # Dimensions
    rows_relax_soc::Vector{Vector{Int}} # Row indices in conic relaxation
    rows_sub_soc::Vector{Vector{Int}} # Row indices in subproblem
    vars_soc::Vector{Vector{JuMP.Variable}} # Slack variables (newly added or detected)
    vars_dagg_soc::Vector{Vector{JuMP.Variable}} # Disaggregated variables
    coefs_soc::Vector{Vector{Float64}} # Coefficients associated with slacks
    isslacknew_soc::Vector{Vector{Bool}} # Indicators for which slacks were newly added

    # Exp data
    num_exp::Int                # Number of ExpPrimal cones
    summ_exp::Dict{Symbol,Real} # Data and infeasibilities
    rows_relax_exp::Vector{Vector{Int}} # Row indices in conic relaxation
    rows_sub_exp::Vector{Vector{Int}} # Row indices in subproblem
    vars_exp::Vector{Vector{JuMP.Variable}} # Slack variables (newly added or detected)
    coefs_exp::Vector{Vector{Float64}} # Coefficients associated with slacks
    isslacknew_exp::Vector{Vector{Bool}} # Indicators for which slacks were newly added

    # SDP data
    num_sdp::Int                # Number of SDP cones
    summ_sdp::Dict{Symbol,Real} # Data and infeasibilities
    rows_relax_sdp::Vector{Vector{Int}} # Row indices in conic relaxation
    rows_sub_sdp::Vector{Vector{Int}} # Row indices in subproblem
    dim_sdp::Vector{Int}        # Dimensions
    vars_svec_sdp::Vector{Vector{JuMP.Variable}} # Slack variables in svec form (newly added or detected)
    coefs_svec_sdp::Vector{Vector{Float64}} # Coefficients associated with slacks in svec form
    vars_smat_sdp::Vector{Array{JuMP.Variable,2}} # Slack variables in smat form (newly added or detected)
    coefs_smat_sdp::Vector{Array{Float64,2}} # Coefficients associated with slacks in smat form
    isslacknew_sdp::Vector{Vector{Bool}} # Indicators for which slacks were newly added
    smat_sdp::Vector{Array{Float64,2}} # Preallocated matrix to help with memory for SDP cut generation

    # Dynamic solve data
    oa_started::Bool            # Indicator for Iterative or MIP-solver-driven algorithms started
    status::Symbol              # Current solve status
    mip_obj::Float64            # Latest MIP (outer approx) objective value
    best_obj::Float64           # Best feasible objective value
    best_int::Vector{Float64}   # Best feasible integer solution
    best_sub::Vector{Float64}   # Best feasible continuous solution
    slck_sub::Vector{Float64}   # Best feasible slack vector (for calculating MIP solution)
    gap_rel_opt::Float64        # Relative optimality gap = |mip_obj - best_obj|/|best_obj|
    cb_heur                     # Heuristic callback reference (MIP-driven only)
    cb_lazy                     # Lazy callback reference (MIP-driven only)

    # Model constructor
    function PajaritoConicModel(log_level, mip_solver_drives, pass_mip_sols, round_mip_sols, mip_subopt_count, mip_subopt_solver, soc_in_mip, disagg_soc, soc_ell_one, soc_ell_inf, exp_init, proj_dual_infeas, proj_dual_feas, viol_cuts_only, mip_solver, cont_solver, timeout, rel_gap, detect_slacks, slack_tol_order, zero_tol, primal_cuts_only, primal_cuts_always, primal_cuts_assist, primal_cut_zero_tol, primal_cut_inf_tol, sdp_init_lin, sdp_init_soc, sdp_eig, sdp_soc, sdp_tol_eigvec, sdp_tol_eigval)
        # Errors
        if soc_in_mip || sdp_init_soc || sdp_soc
            # If using MISOCP outer approximation, check MIP solver handles MISOCP
            mip_spec = MathProgBase.supportedcones(mip_solver)
            if !(:SOC in mip_spec)
                error("The MIP solver specified does not support MISOCP\n")
            end
        end
        if primal_cuts_only && !primal_cuts_always
            # If only using primal cuts, have to use them always
            error("When using primal cuts only, they must be added always\n")
        end
        if (primal_cuts_only || primal_cuts_assist) && (primal_cut_zero_tol < 1e-5)
            # If using primal cuts, cut zero tolerance needs to be sufficiently positive
            error("When using primal cuts, primal cut zero tolerance must be at least 1e-5 to avoid numerical issues\n")
        end

        # Warnings
        if sdp_soc && mip_solver_drives
            warn("SOC cuts for SDP cones cannot be added during the MIP-solver-driven algorithm, but initial SOC cuts may be used\n")
        end
        if mip_solver_drives
            warn("For the MIP-solver-driven algorithm, optimality tolerance must be specified as MIP solver option, not Pajarito option\n")
        end
        if round_mip_sols
            warn("Integer solutions will be rounded: if this seems to cause numerical challenges, change round_mip_sols option\n")
        end
        if primal_cuts_only
            warn("Using primal cuts only may cause convergence issues\n")
        end

        # Initialize model
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.pass_mip_sols = pass_mip_sols
        m.round_mip_sols = round_mip_sols
        m.mip_subopt_count = mip_subopt_count
        m.mip_subopt_solver = mip_subopt_solver
        m.soc_in_mip = soc_in_mip
        m.disagg_soc = disagg_soc
        m.soc_ell_one = soc_ell_one
        m.soc_ell_inf = soc_ell_inf
        m.exp_init = exp_init
        m.proj_dual_infeas = proj_dual_infeas
        m.proj_dual_feas = proj_dual_feas
        m.viol_cuts_only = viol_cuts_only
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.detect_slacks = detect_slacks
        m.slack_tol_order = slack_tol_order
        m.zero_tol = zero_tol
        m.primal_cuts_only = primal_cuts_only
        m.primal_cuts_always = primal_cuts_always
        m.primal_cuts_assist = primal_cuts_assist
        m.primal_cut_zero_tol = primal_cut_zero_tol
        m.primal_cut_inf_tol = primal_cut_inf_tol
        m.sdp_init_lin = sdp_init_lin
        m.sdp_init_soc = sdp_init_soc
        m.sdp_eig = sdp_eig
        m.sdp_soc = sdp_soc
        m.sdp_tol_eigvec = sdp_tol_eigvec
        m.sdp_tol_eigval = sdp_tol_eigval

        m.var_types = Symbol[]
        # m.var_start = Float64[]
        m.num_var_orig = 0
        m.num_con_orig = 0

        m.oa_started = false
        m.best_obj = Inf
        m.best_int = Float64[]
        m.best_sub = Float64[]
        m.status = :NotLoaded

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

    # Verify cone compatibility with solver if solver is not defaultConicsolver
    # TODO defaultConicsolver is an MPB issue
    if m.cont_solver != MathProgBase.defaultConicsolver
        conic_spec = MathProgBase.supportedcones(m.cont_solver)
        for (spec, inds) in vcat(cone_con, cone_var)
            if !(spec in conic_spec)
                if (spec == :SOCRotated) && (:SOC in conic_spec)
                    nothing
                else
                    error("Cones $spec are not supported by the specified conic solver\n")
                end
            end
        end
    end

    # Verify consistency of cone indices and create cone summary dictionary with min/max dimensions of each species
    for (spec, inds) in vcat(cone_con, cone_var)
        # Verify dimensions of cones
        if isempty(inds)
            error("A cone $spec has no associated indices\n")
        end
        if spec == :SOC && (length(inds) < 2)
            error("A cone $spec has fewer than 2 indices ($(length(inds)))\n")
        elseif spec == :SOCRotated && (length(inds) < 3)
            error("A cone $spec has fewer than 3 indices ($(length(inds)))\n")
        elseif spec == :SDP
            if length(inds) < 3
                error("A cone $spec has fewer than 3 indices ($(length(inds)))\n")
            else
                if floor(sqrt(8 * length(inds) + 1)) != sqrt(8 * length(inds) + 1)
                    error("A cone $spec (in SD svec form) does not have a valid (triangular) number of indices ($(length(inds)))\n")
                end
            end
        elseif spec == :ExpPrimal && (length(inds) != 3)
            error("A cone $spec does not have exactly 3 indices ($(length(inds)))\n")
        end
    end

    # Check for values in A smaller than zero tolerance
    A_sp = sparse(A)
    A_num_zeros = count(val -> (abs(val) < m.zero_tol), nonzeros(A_sp))
    if A_num_zeros > 0
        warn("Matrix A has $(A_num_zeros) entries smaller than zero tolerance $(m.zero_tol); performance may be improved by first fixing small magnitudes to zero\n")
    end

    # This is for testing only: set near zeros in A matrix to zero
    # A_sp[abs(A_sp) .< m.zero_tol] = 0.
    # dropzeros!(A_sp)
    # A_sp = sparse(A_sp)

    m.num_con_orig = num_con_orig
    m.num_var_orig = num_var_orig
    m.A_orig = A_sp
    m.c_orig = c
    m.b_orig = b
    m.cone_con_orig = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_con]
    m.cone_var_orig = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_var]
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
function MathProgBase.setvartype!(m::PajaritoConicModel, types_var::Vector{Symbol})
    # Check if vector can be loaded
    if m.status != :Loaded
        error("Must specify variable types right after loading problem\n")
    end
    if length(types_var) != m.num_var_orig
        error("Variable types vector length ($(length(types_var))) does not match number of variables ($(m.num_var_orig))\n")
    end
    if any((var_type -> !(var_type in (:Bin, :Int, :Cont))), types_var)
        error("Some variable types are not in :Bin, :Int, :Cont\n")
    end
    if !any((var_type -> var_type in (:Bin, :Int)), types_var)
        error("No variables are in :Bin, :Int; use conic solver directly if problem is continuous\n")
    end

    m.var_types = types_var
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    # Initialize
    if m.status != :Loaded
        error("Must call optimize! function after loading problem\n")
    end
    if isempty(m.var_types)
        error("Variable types were not specified; must call setvartype! function\n")
    end
    logs = create_logs()

    # Generate model data and instantiate MIP model
    logs[:total] = time()
    trans_data!(m, logs)
    create_conic_data!(m, logs)
    create_mip_data!(m, logs)
    print_cones(m)
    reset_cone_summary!(m)

    # Solve relaxed conic problem, proceed with algorithm if feasible, else finish
    dual_relax = process_relax!(m, logs)
    if !isempty(dual_relax)
        # Add initial dual cuts to MIP model and print info
        add_dual_cuts!(m, dual_relax, true, logs)
        print_inf_dual(m)

        # Begin selected algorithm
        logs[:oa_alg] = time()
        m.oa_started = true
        if m.mip_solver_drives
            # Solve with MIP solver driven outer approximation algorithm
            if m.log_level > 0
                @printf "\nStarting MIP-solver-driven outer approximation algorithm:\n"
            end
            solve_mip_driven!(m, logs)
        else
            # Solve with iterative outer approximation algorithm
            if m.log_level > 0
                @printf "\nStarting iterative outer approximation algorithm:\n"
            end
            solve_iterative!(m, logs)
        end
        logs[:oa_alg] = time() - logs[:oa_alg]
    end

    # Print summary
    logs[:total] = time() - logs[:total]
    print_finish(m, logs)
end

MathProgBase.numconstr(m::PajaritoConicModel) = m.num_con_orig

MathProgBase.numvar(m::PajaritoConicModel) = m.num_var_orig

MathProgBase.status(m::PajaritoConicModel) = m.status

MathProgBase.getobjval(m::PajaritoConicModel) = m.best_obj

MathProgBase.getobjbound(m::PajaritoConicModel) = m.mip_obj

function MathProgBase.getsolution(m::PajaritoConicModel)
    soln = zeros(m.num_var_orig)
    soln_new = zeros(m.num_var_new)
    soln_new[m.cols_int] = m.best_int
    soln_new[m.cols_cont] = m.best_sub
    soln[m.keep_cols] = soln_new
    return soln
end


#=========================================================
 Model constructor functions
=========================================================#

# Transform data: convert variable cones to constraint cones, detect existing slack variables
function trans_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Convert nonlinear variable cones to constraint cones by adding new rows
    cone_con_new = m.cone_con_orig
    cone_var_new = Tuple{Symbol,Vector{Int}}[]
    b_new = m.b_orig
    num_con_new = m.num_con_orig
    num_var_new = 0
    (A_I, A_J, A_V) = findnz(m.A_orig)
    old_new_col = zeros(Int, m.num_var_orig)

    for (spec, cols) in m.cone_var_orig
        if spec == :Zero
            nothing
        elseif spec in (:Free, :NonNeg, :NonPos)
            old_new_col[cols] = collect((num_var_new + 1):(num_var_new + length(cols)))
            push!(cone_var_new, (spec, old_new_col[cols]))
            num_var_new += length(cols)
        else
            old_new_col[cols] = collect((num_var_new + 1):(num_var_new + length(cols)))
            push!(cone_var_new, (:Free, old_new_col[cols]))
            num_var_new += length(cols)

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

    A_zeros = sparse(A_I, A_J, A_V, num_con_new, m.num_var_orig)
    keep_cols = find(old_new_col)
    (A_I, A_J, A_V) = findnz(A_zeros[:, keep_cols])

    # Convert SOCRotated cones to SOC cones
    # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt(2)*x) in SOC, y >= 0, z >= 0
    socr_rows = Vector{Int}[]
    for n_cone in 1:length(cone_con_new)
        (spec, rows) = cone_con_new[n_cone]
        if spec == :SOCRotated
            cone_con_new[n_cone] = (:SOC, rows)
            push!(socr_rows, rows)
        end
    end

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

        # Use old constraint cone SOCRotated for (y+z,-y+z,sqrt(2)*x) in SOC
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
                A_V[ind] *= sqrt(2)
            end
        end
        b_new[rows[2:end]] .*= sqrt(2)
    end

    A_new = sparse(A_I, A_J, A_V, num_con_new, num_var_new)
    (A_I, A_J, A_V) = findnz(A_new)

    # Set up for detecting existing slack variables in nonlinear cone rows with b=0, corresponding to isolated row nonzeros
    row_slck_count = zeros(Int, num_con_new)
    for (ind, i) in enumerate(A_I)
        if (b_new[i] == 0.) && (A_V[ind] != 0.)
            if row_slck_count[i] == 0
                row_slck_count[i] = ind
            elseif row_slck_count[i] > 0
                row_slck_count[i] = -1
            end
        end
    end

    row_to_slckj = Dict{Int,Int}()
    row_to_slckv = Dict{Int,Float64}()

    for (spec, rows) in cone_con_new
        if !(spec in (:Free, :Zero, :NonNeg, :NonPos))
            # If option to detect slacks is true, auto-detect slacks depending on order of magnitude tolerance for abs of the coefficient
            if m.detect_slacks
                if m.slack_tol_order < 0.
                    # Negative slack tol order means only choose -1 coefficients
                    for i in rows
                        if row_slck_count[i] > 0
                            if A_V[row_slck_count[i]] == -1.
                                row_to_slckj[i] = A_J[row_slck_count[i]]
                                row_to_slckv[i] = A_V[row_slck_count[i]]
                            end
                        end
                    end
                elseif m.slack_tol_order == 0.
                    # Zero slack tol order means only choose -1, +1 coefficients
                    for i in rows
                        if row_slck_count[i] > 0
                            if abs(A_V[row_slck_count[i]]) == 1.
                                row_to_slckj[i] = A_J[row_slck_count[i]]
                                row_to_slckv[i] = A_V[row_slck_count[i]]
                            end
                        end
                    end
                else
                    # Positive slack tol order means choose coefficients with abs in order of magnitude range
                    for i in rows
                        if row_slck_count[i] > 0
                            if abs(log10(abs(A_V[row_slck_count[i]]))) <= m.slack_tol_order
                                row_to_slckj[i] = A_J[row_slck_count[i]]
                                row_to_slckv[i] = A_V[row_slck_count[i]]
                            end
                        end
                    end
                end
            end
        end
    end

    # Store transformed data
    m.cone_con_new = cone_con_new
    m.cone_var_new = cone_var_new
    m.num_con_new = num_con_new
    m.num_var_new = num_var_new
    m.c_new = m.c_orig[keep_cols]
    m.keep_cols = keep_cols
    m.b_new = b_new
    m.A_new = A_new
    m.row_to_slckj = row_to_slckj
    m.row_to_slckv = row_to_slckv

    logs[:data_trans] += toq()
end

# Create conic subproblem data by removing integer variable columns and rows without continuous variables
function create_conic_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Build new subproblem variable cones by removing integer variables
    cols_cont = Int[]
    cols_int = Int[]
    num_cont = 0
    cone_var_sub = Tuple{Symbol,Vector{Int}}[]

    for (spec, cols) in m.cone_var_new
        cols_cont_new = Int[]
        for j in cols
            if m.var_types[m.keep_cols][j] == :Cont
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
    (A_cont_I, _, A_cont_V) = findnz(m.A_new[:, cols_cont])
    rows_nz = falses(m.num_con_new)
    for (i, v) in zip(A_cont_I, A_cont_V)
        if !rows_nz[i] && (v != 0)
            rows_nz[i] = true
        end
    end

    # Build new subproblem constraint cones by removing empty rows
    num_full = 0
    rows_full = Int[]
    cone_con_sub = Tuple{Symbol,Vector{Int}}[]
    map_rows_sub = Vector{Int}(length(m.b_new))

    for (spec, rows) in m.cone_con_new
        if spec in (:Zero, :NonNeg, :NonPos)
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
    m.map_rows_sub = map_rows_sub
    m.cols_cont = cols_cont
    m.cols_int = cols_int

    # Build new subproblem A, b, c data by removing empty rows and integer variables
    m.A_sub_cont = m.A_new[rows_full, cols_cont]
    m.A_sub_int = m.A_new[rows_full, cols_int]
    m.b_sub = m.b_new[rows_full]
    m.c_sub_cont = m.c_new[cols_cont]
    m.c_sub_int = m.c_new[cols_int]
    m.b_sub_int = zeros(length(rows_full))
    m.slck_sub = zeros(length(rows_full))

    logs[:data_conic] += toq()
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Initialize JuMP model for MIP outer approximation problem
    model_mip = JuMP.Model(solver=m.mip_solver)

    # Create variables and set types
    x_mip = @variable(model_mip, [1:m.num_var_new])
    for j in 1:m.num_var_new
        setcategory(x_mip[j], m.var_types[m.keep_cols][j])
    end

    # Set objective function
    @objective(model_mip, :Min, dot(m.c_new, x_mip))

    # Add variable cones to MIP
    for (spec, cols) in m.cone_var_new
        if spec == :NonNeg
            for j in cols
                setname(x_mip[j], "v$(j)")
                setlowerbound(x_mip[j], 0.)
            end
        elseif spec == :NonPos
            for j in cols
                setname(x_mip[j], "v$(j)")
                setupperbound(x_mip[j], 0.)
            end
        elseif spec == :Free
            for j in cols
                setname(x_mip[j], "v$(j)")
            end
        elseif spec == :Zero
            error("Bug: Zero cones should have been removed by transform data function (submit an issue)\n")
        end
    end

    # Loop through nonlinear cones to count and summarize
    num_soc = 0
    num_exp = 0
    num_sdp = 0
    summ_soc = Dict{Symbol,Real}(:max_dim => 0, :min_dim => 0)
    summ_exp = Dict{Symbol,Real}(:max_dim => 3, :min_dim => 3)
    summ_sdp = Dict{Symbol,Real}(:max_dim => 0, :min_dim => 0)
    temp_sdp_smat = Dict{Int,Array{Float64,2}}()

    for (spec, rows) in m.cone_con_new
        if spec == :SOC
            num_soc += 1
            if summ_soc[:max_dim] < length(rows)
                summ_soc[:max_dim] = length(rows)
            end
            if (summ_soc[:min_dim] == 0) || (summ_soc[:min_dim] > length(rows))
                summ_soc[:min_dim] = length(rows)
            end
        elseif spec == :ExpPrimal
            num_exp += 1
        elseif spec == :SDP
            num_sdp += 1
            dim = round(Int, sqrt(1/4 + 2 * length(rows)) - 1/2) # smat space dimension
            if summ_sdp[:max_dim] < dim
                summ_sdp[:max_dim] = dim
            end
            if (summ_sdp[:min_dim] == 0) || (summ_sdp[:min_dim] > dim)
                summ_sdp[:min_dim] = dim
            end

            # Preallocate smat matrix for SDP cut functions
            if !haskey(temp_sdp_smat, dim)
                temp_sdp_smat[dim] = Array{Float64,2}(dim, dim)
            end
        end
    end

    # Allocate data for nonlinear cones
    if num_soc > 0
        dim_soc = Vector{Int}(num_soc)
        rows_relax_soc = Vector{Vector{Int}}(num_soc)
        rows_sub_soc = Vector{Vector{Int}}(num_soc)
        vars_soc = Vector{Vector{JuMP.Variable}}(num_soc)
        vars_dagg_soc = Vector{Vector{JuMP.Variable}}(num_soc)
        coefs_soc = Vector{Vector{Float64}}(num_soc)
        isslacknew_soc = Vector{Vector{Bool}}(num_soc)
    end
    if num_exp > 0
        rows_relax_exp = Vector{Vector{Int}}(num_exp)
        rows_sub_exp = Vector{Vector{Int}}(num_exp)
        vars_exp = Vector{Vector{JuMP.Variable}}(num_exp)
        coefs_exp = Vector{Vector{Float64}}(num_exp)
        isslacknew_exp = Vector{Vector{Bool}}(num_exp)
    end
    if num_sdp > 0
        rows_sub_sdp = Vector{Vector{Int}}(num_sdp)
        rows_relax_sdp = Vector{Vector{Int}}(num_sdp)
        dim_sdp = Vector{Int}(num_sdp)
        vars_svec_sdp = Vector{Vector{JuMP.Variable}}(num_sdp)
        coefs_svec_sdp = Vector{Vector{Float64}}(num_sdp)
        vars_smat_sdp = Vector{Array{JuMP.Variable,2}}(num_sdp)
        coefs_smat_sdp = Vector{Array{Float64,2}}(num_sdp)
        isslacknew_sdp = Vector{Vector{Bool}}(num_sdp)
        smat_sdp = Vector{Array{Float64,2}}(num_sdp)
    end

    # Set up a SOC cone in the MIP
    function add_soc!(n_soc, len, rows, vars, coefs, isslacknew)
        dim_soc[n_soc] = len
        rows_relax_soc[n_soc] = rows
        rows_sub_soc[n_soc] = m.map_rows_sub[rows]
        vars_soc[n_soc] = vars
        vars_dagg_soc[n_soc] = Vector{JuMP.Variable}(0)
        coefs_soc[n_soc] = coefs
        isslacknew_soc[n_soc] = isslacknew

        # Set bounds
        if sign(coefs[1]) == 1
            setlowerbound(vars[1], 0.)
        else
            setupperbound(vars[1], 0.)
        end

        # Set names
        for j in 1:len
            setname(vars[j], "s$(j)_soc$(n_soc)")
        end

        # Add constraints depending on options
        if m.soc_in_mip
            # TODO use norm, fix jump issue 784 so that warm start works
            # @constraint(model_mip, norm2{coefs[j] .* vars[j], j in 2:len} <= coefs[1] * vars[1])
            error("SOC in MIP option is currently broken\n")
        elseif m.disagg_soc
            # Add disaggregated SOC variables
            # 2*d_j >= y_j^2/x
            vars_dagg = @variable(model_mip, [j in 1:(len - 1)], lowerbound=0.)
            vars_dagg_soc[n_soc] = vars_dagg

            # Add disaggregated SOC constraint
            # x >= sum(2*d_j)
            @constraint(model_mip, coefs[1] * vars[1] >= 2. * sum(vars_dagg))

            # Set names
            for j in 1:(len - 1)
                setname(vars_dagg[j], "d$(j+1)_soc$(n_soc)")
            end

            # Add initial SOC linearizations
            if m.soc_ell_one
                # Add initial L_1 SOC cuts
                # 2*d_j >= 2*|y_j|/sqrt(len - 1) - x/(len - 1)
                # for all j, implies x*sqrt(len - 1) >= sum(|y_j|)
                # linearize y_j^2/x at x = 1, y_j = 1/sqrt(len - 1) for all j
                for j in 2:len
                    @constraint(model_mip, 2. * vars_dagg[j-1] >=  2. / sqrt(len - 1) * coefs[j] * vars[j] - 1. / (len - 1) * coefs[1] * vars[1])
                    @constraint(model_mip, 2. * vars_dagg[j-1] >= -2. / sqrt(len - 1) * coefs[j] * vars[j] - 1. / (len - 1) * coefs[1] * vars[1])
                end
            end
            if m.soc_ell_inf
                # Add initial L_inf SOC cuts
                # 2*d_j >= 2|y_j| - x
                # implies x >= |y_j|, for all j
                # linearize y_j^2/x at x = 1, y_j = 1 for each j (y_k = 0 for k != j)
                # equivalent to standard 3-dim rotated SOC linearizations x + d_j >= 2|y_j|
                for j in 2:len
                    @constraint(model_mip, 2. * vars_dagg[j-1] >=  2. * coefs[j] * vars[j] - coefs[1] * vars[1])
                    @constraint(model_mip, 2. * vars_dagg[j-1] >= -2. * coefs[j] * vars[j] - coefs[1] * vars[1])
                end
            end
        end
    end

    # Set up a ExpPrimal cone in the MIP
    function add_exp!(n_exp, rows, vars, coefs, isslacknew)
        rows_relax_exp[n_exp] = rows
        rows_sub_exp[n_exp] = m.map_rows_sub[rows]
        vars_exp[n_exp] = vars
        coefs_exp[n_exp] = coefs
        isslacknew_exp[n_exp] = isslacknew

        # Set bounds
        if sign(coefs[2]) == 1
            setlowerbound(vars[2], 0.)
        else
            setupperbound(vars[2], 0.)
        end
        if sign(coefs[3]) == 1
            setlowerbound(vars[3], 0.)
        else
            setupperbound(vars[3], 0.)
        end

        # Set names
        for j in 1:3
            setname(vars[j], "s$(j)_exp$(n_exp)")
        end

        # Add initial linearization depending on option
        if m.exp_init
            # TODO maybe pick different linearization points
            # Add initial exp cuts using dual exp cone linearizations
            # Dual exp cone is  e * z >= -x * exp(y / x), z >= 0, x < 0
            # at x = -1; y = -1, -1/2, -1/5, 0, 1/5, 1/2, 1; z = exp(-y) / e = exp(-y - 1)
            for yval in [-1., -0.5, -0.2, 0., 0.2, 0.5, 1.]
                @constraint(model_mip, -coefs[1] * vars[1] + yval * coefs[1] * vars[1] + exp(-yval - 1.) * coefs[3] * vars[3] >= 0)
            end
        end
    end

    # Set up a SDP cone in the MIP
    function add_sdp!(n_sdp, dim, rows, vars, coefs, isslacknew)
        dim_sdp[n_sdp] = dim
        rows_relax_sdp[n_sdp] = rows
        rows_sub_sdp[n_sdp] = m.map_rows_sub[rows]
        vars_svec_sdp[n_sdp] = vars
        coefs_svec_sdp[n_sdp] = coefs
        isslacknew_sdp[n_sdp] = isslacknew
        smat_sdp[n_sdp] = temp_sdp_smat[dim]
        vars_smat_sdp[n_sdp] = vars_smat = Array{JuMP.Variable,2}(dim, dim)
        coefs_smat_sdp[n_sdp] = coefs_smat = Array{Float64,2}(dim, dim)

        # Set up smat arrays and set bounds and names
        kSD = 1
        for jSD in 1:dim, iSD in jSD:dim
            setname(vars[kSD], "s$(kSD)_smat($(iSD),$(jSD))_sdp$(n_sdp)")
            vars_smat[iSD, jSD] = vars_smat[jSD, iSD] = vars[kSD]
            if jSD == iSD
                coefs_smat[iSD, jSD] = coefs_smat[jSD, iSD] = coefs[kSD]
                if sign(coefs[kSD]) == 1
                    setlowerbound(vars[kSD], 0.)
                else
                    setupperbound(vars[kSD], 0.)
                end
            else
                # Detect if slack coefficient is sqrt(2) ie smat coefficient is 1
                if abs(coefs[kSD]) == sqrt(2)
                    coefs_smat[iSD, jSD] = coefs_smat[jSD, iSD] = sign(coefs[kSD])
                else
                    coefs_smat[iSD, jSD] = coefs_smat[jSD, iSD] = coefs[kSD] / sqrt(2)
                end
            end
            kSD += 1
        end

        # Add initial SDP linear constraints
        # TODO these are redundant with init SOCs so don't do both
        if m.sdp_init_lin
            for jSD in 1:dim, iSD in (jSD + 1):dim
                # Add initial SDP linear cuts based on linearization of 3-dim rotated SOCs that enforce 2x2 principal submatrix PSDness (essentially the dual of SDSOS)
                # 2|m_ij| <= m_ii + m_jj, where m_kk is scaled by sqrt(2) in smat space
                @constraint(model_mip, coefs_smat[iSD, iSD] * vars_smat[iSD, iSD] + coefs_smat[jSD, jSD] * vars_smat[jSD, jSD] >= 2. * coefs_smat[iSD, jSD] * vars_smat[iSD, jSD])
                @constraint(model_mip, coefs_smat[iSD, iSD] * vars_smat[iSD, iSD] + coefs_smat[jSD, jSD] * vars_smat[jSD, jSD] >= -2. * coefs_smat[iSD, jSD] * vars_smat[iSD, jSD])
            end
        end

        # Add initial SDP SOC cuts
        if m.sdp_init_soc || m.sdp_soc
            error("SOC in MIP is currently broken\n")
            # TODO maybe fix jump issue 784 so that warm start works
            # help = @variable(model_mip, [j in 1:dim], lowerbound=0., basename="h$(n_sdp)SDhelp")
            # kSD = 1
            # for jSD in 1:dim, iSD in jSD:dim
            #     if jSD == iSD
            #         # Create helper variable corresponding to diagonal element multiplied by sqrt(2)
            #         @constraint(model_mip, help[jSD] == coefs[kSD] * sqrt(2) * vars[kSD])
            #         smat[jSD, iSD] = help[jSD]
            #     else
            #         smat[jSD, iSD] = smat[iSD, jSD] = coefs[kSD] * vars[kSD]
            #         if m.sdp_init_soc
            #             # Add initial rotated SOC for off-diagonal element to enforce 2x2 principal submatrix PSDness
            #             # TODO this won't work: not intepreted as rsoc, may need manual SOC transformation, norm maybe?
            #             @constraint(model_mip, help[jSD] * help[iSD] >= (coefs[kSD] * vars[kSD])^2)
            #         end
            #     end
            #     kSD += 1
            # end
        end
    end

    n_soc = 0
    n_exp = 0
    n_sdp = 0

    lhs_expr = m.b_new - m.A_new * x_mip

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    for (spec, rows) in m.cone_con_new
        if spec == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0.)
        elseif spec == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)
        elseif spec == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)
        else
            # Set up nonlinear cone slacks and data
            len = length(rows)
            vars = Vector{JuMP.Variable}(len)
            coefs = ones(len)
            isslacknew = Vector{Bool}(len)

            for (ind, i) in enumerate(rows)
                if haskey(m.row_to_slckj, i)
                    vars[ind] = x_mip[m.row_to_slckj[i]]
                    coefs[ind] = - m.row_to_slckv[i]
                    isslacknew[ind] = false
                else
                    vars[ind] = @variable(model_mip, _)
                    @constraint(model_mip, lhs_expr[i] - vars[ind] == 0.)
                    isslacknew[ind] = true
                end
            end

            # Set up MIP cones
            if spec == :SOC
                n_soc += 1
                add_soc!(n_soc, len, rows, vars, coefs, isslacknew)
            elseif spec == :ExpPrimal
                n_exp += 1
                add_exp!(n_exp, rows, vars, coefs, isslacknew)
            elseif spec == :SDP
                n_sdp += 1
                dim = round(Int, sqrt(1/4 + 2 * len) - 1/2) # smat space dimension
                add_sdp!(n_sdp, dim, rows, vars, coefs, isslacknew)
            end
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_mip = x_mip

    m.num_soc = num_soc
    if num_soc > 0
        m.summ_soc = summ_soc
        m.dim_soc = dim_soc
        m.rows_sub_soc = rows_sub_soc
        m.rows_relax_soc = rows_relax_soc
        m.vars_soc = vars_soc
        m.vars_dagg_soc = vars_dagg_soc
        m.coefs_soc = coefs_soc
        m.isslacknew_soc = isslacknew_soc
    end
    m.num_exp = num_exp
    if num_exp > 0
        m.summ_exp = summ_exp
        m.rows_relax_exp = rows_relax_exp
        m.rows_sub_exp = rows_sub_exp
        m.vars_exp = vars_exp
        m.coefs_exp = coefs_exp
        m.isslacknew_exp = isslacknew_exp
    end
    m.num_sdp = num_sdp
    if num_sdp > 0
        m.summ_sdp = summ_sdp
        m.rows_relax_sdp = rows_relax_sdp
        m.rows_sub_sdp = rows_sub_sdp
        m.dim_sdp = dim_sdp
        m.vars_svec_sdp = vars_svec_sdp
        m.coefs_svec_sdp = coefs_svec_sdp
        m.vars_smat_sdp = vars_smat_sdp
        m.coefs_smat_sdp = coefs_smat_sdp
        m.isslacknew_sdp = isslacknew_sdp
        m.smat_sdp = smat_sdp
    end

    logs[:data_mip] += toq()
    # println(model_mip)
end


#=========================================================
 Iterative algorithm functions
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    cache_soln = Set{Vector{Float64}}()
    soln_int = Vector{Float64}(length(m.cols_int))
    count_early = m.mip_subopt_count

    while true
        # Reset cones summary values
        reset_cone_summary!(m)

        if count_early < m.mip_subopt_count
            # Solve is a partial solve: use subopt MIP solver
            # Don't set time limit: trust that user has provided reasonably small time limit
            setsolver(m.model_mip, m.mip_subopt_solver)
        else
            # Solve is a full solve: use full MIP solver with remaining time limit
            if applicable(MathProgBase.setparameters!, m.mip_solver)
                MathProgBase.setparameters!(m.mip_solver, TimeLimit=(m.timeout - (time() - logs[:total])))
            end
            setsolver(m.model_mip, m.mip_solver)
        end

        # Solve MIP
        tic()
        status_mip = solve(m.model_mip)
        logs[:mip_solve] += toq()

        # Use MIP status
        if status_mip in (:Infeasible, :InfeasibleOrUnbounded)
            # Stop if infeasible
            m.status = :Infeasible
            break
        elseif status_mip == :Unbounded
            # Stop if unbounded (initial conic relax solve should detect this)
            warn("MIP solver returned status $status_mip, which could indicate that the initial dual cuts added were too weak: aborting iterative algorithm\n")
            m.status = :MIPFailure
            break
        elseif status_mip in (:UserLimit, :Suboptimal)
            # MIP stopped early, so check if OA bound is higher than current best bound; increment early mip solve count
            if applicable(MathProgBase.getobjbound, m.model_mip)
                mip_obj_bound = MathProgBase.getobjbound(m.model_mip)
                if mip_obj_bound > m.mip_obj
                    m.mip_obj = mip_obj_bound
                end
            end
            count_early += 1
        elseif status_mip == :Optimal
            # MIP is optimal, so it gives best OA bound; reset early MIP solve count
            m.mip_obj = getobjectivevalue(m.model_mip)
            count_early = 0
        else
            warn("MIP solver returned status $status_mip, which Pajarito does not handle (please submit an issue): aborting iterative algorithm\n")
            m.status = :MIPFailure
            break
        end

        # Get integer solution
        for ind in 1:length(soln_int)
            soln_int[ind] = getvalue(m.x_mip[m.cols_int][ind])
            if isnan(soln_int[ind])
                warn("Integer solution vector has NaN values: aborting iterative algorithm\n")
                m.status = :MIPFailure
                break
            end
        end

        # Round integer solution if option
        if m.round_mip_sols
            for ind in 1:length(soln_int)
                soln_int[ind] = round(soln_int[ind])
            end
        end

        # Check if integer solution has been seen before
        if soln_int in cache_soln
            # Integer solution has repeated: don't call subproblem
            logs[:n_repeat] += 1

            # If MIP was run until optimal
            if count_early == 0
                # Check if converged hence optimal, else return suboptimal or add primal cuts and try again
                m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
                if m.gap_rel_opt < m.rel_gap
                    m.status = :Optimal
                    break
                else
                    # Calculate cone outer infeasibilities of MIP solution, add any violated primal cuts if using primal cuts
                    (oa_viol, cut_viol) = calc_outer_inf_cuts!(m, (m.primal_cuts_always || m.primal_cuts_assist), logs)

                    # If no violated primal cuts were added, finish with suboptimal, else re-solve with the new violated primal cuts
                    if !cut_viol
                        if m.round_mip_sols
                            warn("Rounded integer solutions are cycling before convergence tolerance is reached: aborting iterative algorithm\n")
                        else
                            warn("Non-rounded integer solutions are cycling before convergence tolerance is reached: aborting iterative algorithm\n")
                        end
                        m.status = :Suboptimal
                        break
                    end
                end
            else
                # Calculate cone outer infeasibilities of MIP solution, add any violated primal cuts if always using them
                calc_outer_inf_cuts!(m, m.primal_cuts_always, logs)
            end

            # Run MIP to optimality next iteration
            count_early = m.mip_subopt_count
        else
            # Integer solution is new: save it in the set
            push!(cache_soln, copy(soln_int))

            # Solve conic subproblem, finish if encounter conic solver failure
            (dual_conic, soln_conic) = process_conic!(m, soln_int, logs)
            if isempty(dual_conic)
                m.status = :ConicFailure
                break
            end

            # Update best feasible solution
            if !isempty(soln_conic)
                update_best_soln!(m, soln_int, soln_conic, logs)
            end

            # Calculate relative outer approximation gap, finish if satisfy optimality gap condition
            m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
            if m.gap_rel_opt < m.rel_gap
                m.status = :Optimal
                break
            end

            # Add dual cuts to MIP
            if !m.primal_cuts_only
                add_dual_cuts!(m, dual_conic, false, logs)
            end

            # Calculate cone outer infeasibilities of MIP solution, add any violated primal cuts if always using them
            calc_outer_inf_cuts!(m, m.primal_cuts_always, logs)
        end

        # Finish if exceeded timeout option
        if (time() - logs[:oa_alg]) > m.timeout
            m.status = :UserLimit
            break
        end

        # Give the best feasible solution to the MIP as a warm-start
        # TODO use this at start when enable warm-starting Pajarito
        if m.pass_mip_sols && !isempty(m.best_sub)
            set_best_soln!(m, logs)
        end
    end
end

# Solve the MIP model using MIP-solver-driven callback algorithm
function solve_mip_driven!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    cache_soln = Dict{Vector{Float64},Vector{Float64}}()
    soln_int = Vector{Float64}(length(m.cols_int))
    add_best_soln = false

    # Add lazy cuts callback
    function callback_lazy(cb)
        m.cb_lazy = cb

        # Reset cones summary values
        reset_cone_summary!(m)

        # Get integer solution
        for ind in 1:length(soln_int)
            soln_int[ind] = getvalue(m.x_mip[m.cols_int][ind])
            # if isnan(soln_int[ind])
            #     warn("Integer solution vector has NaN values: aborting MIP-solver-driven algorithm\n")
            #     m.status = :MIPFailure
            #     throw(CallbackAbort())
            # end
        end

        # Round integer solution if option
        if m.round_mip_sols
            for ind in 1:length(soln_int)
                soln_int[ind] = round(soln_int[ind])
            end
        end

        # If integer solution is in cache dictionary
        if haskey(cache_soln, soln_int)
            # Integer solution has been seen before
            logs[:n_repeat] += 1

            # Calculate cone outer infeasibilities of MIP solution, add any violated primal cuts if using primal cuts
            (oa_viol, cut_viol) = calc_outer_inf_cuts!(m, (m.primal_cuts_always || m.primal_cuts_assist), logs)

            # If there are positive outer infeasibilities and no primal cuts were added, add cached dual cuts
            if oa_viol && !cut_viol
                # Get cached conic dual associated with repeated integer solution, re-add all dual cuts
                add_dual_cuts!(m, cache_soln[soln_int], false, logs)
            end
        else
            # Integer solution is new: solve conic subproblem, finish if encounter conic solver failure
            (dual_conic, soln_conic) = process_conic!(m, soln_int, logs)
            if isempty(dual_conic)
                m.status = :ConicFailure
                throw(CallbackAbort())
            end

            # Update best feasible solution: if it changes, update add_best_soln to true to tell heuristic callback to add to MIP
            if !isempty(soln_conic)
                add_best_soln = update_best_soln!(m, soln_int, soln_conic, logs)
            end

            # Cache solution
            if m.primal_cuts_always || m.primal_cuts_assist
                # If using primal cuts, don't need to save dual vector
                cache_soln[copy(soln_int)] = Float64[]
            else
                # Save dual vector associated with integer solution so can re-add dual cuts on repeated integer solutions
                cache_soln[copy(soln_int)] = dual_conic
            end

            # Add dual cuts to MIP
            if !m.primal_cuts_only
                add_dual_cuts!(m, dual_conic, false, logs)
            end

            # Calculate cone outer infeasibilities of MIP solution, add any violated primal cuts if always using them
            calc_outer_inf_cuts!(m, m.primal_cuts_always, logs)
        end
    end

    addlazycallback(m.model_mip, callback_lazy)

    if m.pass_mip_sols
        # Add heuristic callback
        function callback_heur(cb)
            # If have a new best feasible solution since last heuristic solution added
            if add_best_soln
                # Set MIP solution to the new best feasible solution
                m.cb_heur = cb
                set_best_soln!(m, logs)
                addsolution(cb)
                add_best_soln = false
            end
        end

        addheuristiccallback(m.model_mip, callback_heur)
    end

    # Start MIP solver
    logs[:mip_solve] = time()
    if applicable(MathProgBase.setparameters!, m.mip_solver)
        MathProgBase.setparameters!(m.mip_solver, TimeLimit=(m.timeout - (time() - logs[:total])))
        setsolver(m.model_mip, m.mip_solver)
    end
    logs[:mip_solve] = time() - logs[:mip_solve]

    status_mip = solve(m.model_mip)
    if status_mip in (:Infeasible, :InfeasibleOrUnbounded)
        m.status = :Infeasible
    elseif status_mip == :Unbounded
        # Should not be unbounded: initial conic relax solve should detect this
        warn("MIP solver returned status $status_mip, which could indicate that the initial dual cuts added were too weak\n")
        m.status = :MIPFailure
    elseif status_mip in (:UserLimit, :Optimal)
        m.mip_obj = getobjbound(m.model_mip)
        m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
        if !(m.status in (:ConicFailure, :MIPFailure))
            m.status = status_mip
        end
    else
        warn("MIP solver returned status $status_mip, which Pajarito does not handle (please submit an issue)\n")
        m.status = :MIPFailure
    end
end


#=========================================================
 Conic solve functions
=========================================================#

# Solve the initial conic relaxation model
function process_relax!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    # Instantiate and solve the conic relaxation model
    tic()
    model_relax = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_relax, m.c_new, m.A_new, m.b_new, m.cone_con_new, m.cone_var_new)
    MathProgBase.optimize!(model_relax)
    status_relax = MathProgBase.status(model_relax)
    logs[:relax_solve] += toq()

    # Only proceed if status is optimal or suboptimal, and print
    if status_relax == :Infeasible
        warn("Initial conic relaxation status was $status_relax: terminating Pajarito\n")
        m.status = :Infeasible
        return Float64[]
    elseif status_relax == :Unbounded
        warn("Initial conic relaxation status was $status_relax: terminating Pajarito\n")
        m.status = :InfeasibleOrUnbounded
        return Float64[]
    elseif !(status_relax in (:Optimal, :Suboptimal))
        warn("Apparent conic solver failure with status $status_relax: terminating Pajarito\n")
        m.status = :ConicFailure
        return Float64[]
    end

    if m.log_level > 0
        @printf "\nConic relaxation model solved:\n"
        @printf " - Status     = %14s\n" status_relax
        @printf " - Objective  = %14.6f\n" MathProgBase.getobjval(model_relax)
    end

    dual_relax = MathProgBase.getdual(model_relax)

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_relax)
        MathProgBase.freemodel!(model_relax)
    end

    return dual_relax
end

# Solve a conic subproblem given some solution to the integer variables
function process_conic!(m::PajaritoConicModel, soln_int::Vector{Float64}, logs::Dict{Symbol,Real})
    # Calculate new conic constant vector b as m.b_sub_int = m.b_sub - m.A_sub_int*soln_int
    tic()
    A_mul_B!(m.b_sub_int, m.A_sub_int, soln_int) # m.b_sub_int = m.A_sub_int*soln_int
    scale!(m.b_sub_int, -1) # m.b_sub_int = - m.b_sub_int
    BLAS.axpy!(1, m.b_sub, m.b_sub_int) # m.b_sub_int = m.b_sub_int + m.b_sub

    # Load conic model
    # TODO implement changing only RHS vector in MathProgBase
    model_conic = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_conic, m.c_sub_cont, m.A_sub_cont, m.b_sub_int, m.cone_con_sub, m.cone_var_sub)
    logs[:conic_load] += toq()

    # Solve conic model
    tic()
    MathProgBase.optimize!(model_conic)
    logs[:conic_solve] += toq()
    logs[:n_conic] += 1

    # Only proceed if status is infeasible, optimal or suboptimal
    status_conic = MathProgBase.status(model_conic)
    if !(status_conic in (:Optimal, :Suboptimal, :Infeasible))
        if m.mip_solver_drives
            warn("Apparent conic solver failure with status $status_conic: aborting MIP-solver-driven algorithm\n")
        else
            warn("Apparent conic solver failure with status $status_conic: aborting iterative algorithm\n")
        end
        dual_conic = Float64[]
        soln_conic = Float64[]
    else
        dual_conic = MathProgBase.getdual(model_conic)
        if status_conic != :Infeasible
            soln_conic = MathProgBase.getsolution(model_conic)
        else
            soln_conic = Float64[]
        end
    end

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_conic)
        MathProgBase.freemodel!(model_conic)
    end

    return (dual_conic, soln_conic)
end

# Check if solution (integer and conic) has best feasible objective, if so, update and return true
function update_best_soln!(m::PajaritoConicModel, soln_int::Vector{Float64}, soln_cont::Vector{Float64}, logs::Dict{Symbol,Real})
    tic()
    # Calculate objective of feasible solution
    obj_new = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_cont)

    # If have new best feasible solution
    if obj_new <= m.best_obj
        # Save objective and best int and cont variable solutions
        m.best_obj = obj_new
        m.best_int = soln_int
        m.best_sub = soln_cont

        # Calculate and save slack values for use in MIP solution construction
        # m.slck_sub = m.b_sub_int - m.A_sub_cont * m.best_sub
        A_mul_B!(m.slck_sub, m.A_sub_cont, m.best_sub)
        scale!(m.slck_sub, -1)
        BLAS.axpy!(1, m.b_sub_int, m.slck_sub)

        logs[:conic_soln] += toq()
        return true
    else
        logs[:conic_soln] += toq()
        return false
    end
end


#=========================================================
 Dual cuts functions
=========================================================#

# Add dual cuts for each cone and calculate infeasibilities for cuts and duals
function add_dual_cuts!(m::PajaritoConicModel, dual_conic::Vector{Float64}, initbool::Bool, logs::Dict{Symbol,Real})
    tic()
    for n in 1:m.num_soc
        add_dual_cuts_soc!(m, m.dim_soc[n], m.vars_soc[n], m.vars_dagg_soc[n], m.coefs_soc[n], dual_conic[(initbool ? m.rows_relax_soc[n] : m.rows_sub_soc[n])], m.summ_soc)
    end
    for n in 1:m.num_exp
        add_dual_cuts_exp!(m, m.vars_exp[n], m.coefs_exp[n], dual_conic[(initbool ? m.rows_relax_exp[n] : m.rows_sub_exp[n])], m.summ_exp)
    end
    for n in 1:m.num_sdp
        add_dual_cuts_sdp!(m, m.dim_sdp[n], m.vars_smat_sdp[n], m.coefs_smat_sdp[n], dual_conic[(initbool ? m.rows_relax_sdp[n] : m.rows_sub_sdp[n])], m.smat_sdp[n], m.summ_sdp)
    end

    if initbool
        print_inf_dual(m)
    else
        print_inf_dualcuts(m)
    end
    logs[:dual_cuts] += toq()
end

# Add dual cuts for a SOC
function add_dual_cuts_soc!(m::PajaritoConicModel, dim::Int, vars::Vector{JuMP.Variable}, vars_dagg::Vector{JuMP.Variable}, coefs::Vector{Float64}, dual::Vector{Float64}, spec_summ::Dict{Symbol,Real})
    # 0 Rescale by largest absolute value or discard if near zero
    if maxabs(dual) > m.zero_tol
        scale!(dual, (1. / maxabs(dual)))
    else
        return
    end

    # 1 Calculate dual inf
    inf_dual = norm(dual[2:end]) - dual[1]
    if m.log_level > 2
        update_inf_dual!(inf_dual, spec_summ)
    end

    # 2 Sanitize: remove near-zeros
    for ind in 1:dim
        if abs(dual[ind]) < m.zero_tol
            dual[ind] = 0.
        end
    end

    # 2 Project dual if infeasible and proj_dual_infeas or if strictly feasible and proj_dual_feas
    if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
        # Projection: epigraph variable equals norm
        dual[1] = norm(dual[2:end])
    end

    # Discard cut if epigraph variable is 0
    if dual[1] <= 0.
        return
    end

    if m.disagg_soc
        for j in 2:dim
            # TODO are there cases where we don't want to add this? eg if zero
            # 3 Add disaggregated 3-dim cut and update cut infeasibility
            @expression(m.model_mip, cut_expr, (dual[j] / dual[1])^2 * coefs[1] * vars[1] + 2. * vars_dagg[j-1] + (2 * dual[j] / dual[1]) * coefs[j] * vars[j])
            if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
                if m.mip_solver_drives && m.oa_started
                    @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                else
                    @constraint(m.model_mip, cut_expr >= 0.)
                end

                if (m.log_level > 2) && m.oa_started
                    update_inf_cut!(getvalue(cut_expr), spec_summ)
                end
            end
        end
    else
        # 3 Add nondisaggregated cut and update cut infeasibility
        @expression(m.model_mip, cut_expr, sum{dual[j] * coefs[j] * vars[j], j in 1:dim})
        if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
            if m.mip_solver_drives && m.oa_started
                @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
            else
                @constraint(m.model_mip, cut_expr >= 0.)
            end

            if (m.log_level > 2) && m.oa_started
                update_inf_cut!(getvalue(cut_expr), spec_summ)
            end
        end
    end
end

# Add dual cut for a ExpPrimal cone
function add_dual_cuts_exp!(m::PajaritoConicModel, vars::Vector{JuMP.Variable}, coefs::Vector{Float64}, dual::Vector{Float64}, spec_summ::Dict{Symbol,Real})
    # 0 Rescale by largest absolute value or discard if near zero
    if maxabs(dual) > m.zero_tol
        scale!(dual, (1. / maxabs(dual)))
    else
        return
    end

    # 1 Calculate dual inf using exp space definition of dual cone as e * dual[3] >= -dual[1] * exp(dual[2] / dual[1])
    if dual[1] == 0.
        if (dual[2] >= 0.) && (dual[3] >= 0.)
            inf_dual = -max(dual[2], dual[3])
        elseif (dual[2] < 0.) || (dual[3] < 0.)
            inf_dual = max(-dual[2], -dual[3])
        end
    elseif dual[1] > 0.
        inf_dual = dual[1]
    elseif dual[3] < 0.
        inf_dual = -dual[3]
    else
        inf_dual = -dual[1] * exp(dual[2] / dual[1]) - e * dual[3]
    end
    if m.log_level > 2
        update_inf_dual!(inf_dual, spec_summ)
    end

    # 2 Sanitize: remove near-zeros
    for ind in 1:3
        if abs(dual[ind]) < m.zero_tol
            dual[ind] = 0.
        end
    end

    # 2 Project dual if infeasible and proj_dual_infeas or if strictly feasible and proj_dual_feas
    if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
        # Projection: epigraph variable equals LHS
        dual[3] = -dual[1] * exp(dual[2] / dual[1] - 1.)
    end

    # Discard cut if dual[1] >= 0 (simply enforces the nonnegativity of x[2] and x[3]) or dual[3] < 0 (can't project onto dual[3] = 0)
    if (dual[1] >= 0.) || (dual[3] < 0.)
        return
    end

    # 3 Add 3-dim cut
    @expression(m.model_mip, cut_expr, sum{dual[j] * coefs[j] * vars[j], j in 1:3})
    if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
        if m.mip_solver_drives && m.oa_started
            @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
        else
            @constraint(m.model_mip, cut_expr >= 0.)
        end

        if (m.log_level > 2) && m.oa_started
            update_inf_cut!(getvalue(cut_expr), spec_summ)
        end
    end
end

# Add dual cuts for a SDP cone
function add_dual_cuts_sdp!(m::PajaritoConicModel, dim::Int, vars_smat::Array{JuMP.Variable,2}, coefs_smat::Array{Float64,2}, dual::Vector{Float64}, smat::Array{Float64,2}, spec_summ::Dict{Symbol,Real})
    # 0 Rescale by largest absolute value or discard if near zero
    if maxabs(dual) > m.zero_tol
        scale!(dual, (1. / maxabs(dual)))
    else
        return
    end

    # Convert dual to smat space and store in preallocated smat matrix
    make_smat!(dual, smat, dim)

    # Get eigendecomposition of smat dual (use symmetric property), save eigenvectors in smat matrix
    (eigvals, _) = LAPACK.syev!('V', 'L', smat)

    # 1 Calculate dual inf as negative minimum eigenvalue
    inf_dual = -minimum(eigvals)
    if m.log_level > 2
        update_inf_dual!(inf_dual, spec_summ)
    end

    # Discard cut if largest eigenvalue is too small
    if maximum(eigvals) <= m.sdp_tol_eigval
        return
    end

    # 2 Project dual if infeasible and proj_dual_infeas, create cut expression
    if (inf_dual > 0.) && m.proj_dual_infeas
        @expression(m.model_mip, cut_expr, sum{eigvals[v] * smat[vi, v] * smat[vj, v] * (vi == vj ? 1. : 2.) * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim, v in 1:dim; eigvals[v] > 0.})
    else
        @expression(m.model_mip, cut_expr, sum{eigvals[v] * smat[vi, v] * smat[vj, v] * (vi == vj ? 1. : 2.) * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim, v in 1:dim; eigvals[v] != 0.})
    end

    # 3 Add super-rank linear dual cut
    if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
        if m.mip_solver_drives && m.oa_started
            @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
        else
            @constraint(m.model_mip, cut_expr >= 0.)
        end

        if (m.log_level > 2) && m.oa_started
            update_inf_cut!(getvalue(cut_expr), spec_summ)
        end
    end

    if m.sdp_eig
        for v in 1:dim
            if eigvals[v] > m.sdp_tol_eigval
                # 3 Add non-sparse rank-1 cut from smat eigenvector v
                @expression(m.model_mip, cut_expr, sum{(vi == vj ? 1. : 2.) * smat[vi, v] * smat[vj, v] * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim})
                if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
                    if m.mip_solver_drives && m.oa_started
                        @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                    else
                        @constraint(m.model_mip, cut_expr >= 0.)
                    end

                    if (m.log_level > 2) && m.oa_started
                        update_inf_cut!(getvalue(cut_expr), spec_summ)
                    end
                end

                # 2 Sanitize eigenvector v for sparser rank-1 cut
                # TODO try for multiple levels of sparsity
                # TODO these extra cuts slow down MSD, only use for iterative maybe
                for vi in 1:dim
                    if abs(smat[vi, v]) < m.sdp_tol_eigvec
                        smat[vi, v] = 0.
                    end
                end

                # 3 Add sparse rank-1 cut from smat sparsified eigenvector v
                @expression(m.model_mip, cut_expr, sum{(vi == vj ? 1. : 2.) * smat[vi, v] * smat[vj, v] * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim})
                if !m.viol_cuts_only || !m.oa_started || (getvalue(cut_expr) > 0.)
                    if m.mip_solver_drives && m.oa_started
                        @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                    else
                        @constraint(m.model_mip, cut_expr >= 0.)
                    end

                    if (m.log_level > 2) && m.oa_started
                        update_inf_cut!(getvalue(cut_expr), spec_summ)
                    end
                end

                # 3 Add rank-1 Schur complement svec cut for each row index iSD, from smat eigenvector jSD
                # TODO check correctness, see if it helps, make it an option
                # TODO only add side that is violated
                # TODO could instead add rank-n schur cuts. which one dominates??
                # if m.sdp_soclin
                    # for iSD in 1:dim
                    #     # Cut for Schur vector positive
                    #     # Cut for Schur vector negative
                    #       @constraint(m.model_mip, coefs[iSD, iSD] * vars[iSD, iSD] + sum{smat[k, jSD] * smat[l, jSD] * coefs[k, l] * vars[k, l], k in 1:dim, l in 1:dim; k != iSD && l != iSD} >= sum{ 2 * smat[k, iSD] * smat[k, jSD] * coefs[k, iSD] * vars[k, iSD], k in 1:dim; k != iSD})
                    #       @constraint(m.model_mip, coefs[iSD, iSD] * vars[iSD, iSD] + sum{smat[k, jSD] * smat[l, jSD] * coefs[k, l] * vars[k, l], k in 1:dim, l in 1:dim; k != iSD && l != iSD} >= sum{-2 * smat[k, iSD] * smat[k, jSD] * coefs[k, iSD] * vars[k, iSD], k in 1:dim; k != iSD})
                    # end
                # end
            end
        end
    end
end

# Update dual infeasibility values in cone summary
function update_inf_dual!(inf_dual::Float64, spec_summ::Dict{Symbol,Real})
    if inf_dual > 0.
        spec_summ[:dual_max_n] += 1
        spec_summ[:dual_max] = max(inf_dual, spec_summ[:dual_max])
    elseif inf_dual < 0.
        spec_summ[:dual_min_n] += 1
        spec_summ[:dual_min] = max(-inf_dual, spec_summ[:dual_min])
    end
end

# Update cut infeasibility values in cone summary
function update_inf_cut!(inf_cut::Float64, spec_summ::Dict{Symbol,Real})
    if inf_cut > 0.
        spec_summ[:cut_max_n] += 1
        spec_summ[:cut_max] = max(inf_cut, spec_summ[:cut_max])
    elseif inf_cut < 0.
        spec_summ[:cut_min_n] += 1
        spec_summ[:cut_min] = max(-inf_cut, spec_summ[:cut_min])
    end
end

# Add SDP SOC cuts (derived from Schur complement) for each eigenvector and each diagonal element
# TODO broken
# function add_sdp_soc_cuts!(m::PajaritoConicModel, spec_summ::Dict{Symbol,Real}, vars::Tuple{Vector{JuMP.Variable},Vector{JuMP.Variable},Array{JuMP.Variable,2}}, coefs::Vector{Float64}, Vdual::Array{Float64,2})
#     (dim, nV) = size(Vdual)
#
#     # For each sanitized eigenvector with significant eigenvalue
#     for jV in 1:nV
#         # Get eigenvector and form rank-1 outer product
#         vj = Vdual[:, jV]
#         vvj = vj * vj'
#
#         # For each diagonal element of SDP
#         for iSD in 1:dim
#             no_i = vcat(1:(iSD - 1), (iSD + 1):dim)
#
#             # Add helper variable for subvector iSD product
#             @variable(m.model_mip, vx, basename="h$(n)SDvx_$(jV)_$(iSD)")
#             @constraint(m.model_mip, vx == vecdot(vj[no_i], vars[no_i, iSD]))
#
#             # Add helper variable for submatrix iSD product
#             @variable(m.model_mip, vvX >= 0., basename="h$(n)SDvvX_$(jV)_$(iSD)")
#             @constraint(m.model_mip, vvX == vecdot(vvj[no_i, no_i], vars[no_i, no_i]))
#
#             # Add SOC constraint
#             @constraint(m.model_mip, vars[iSD, iSD] * vvX >= vx^2)
#
#             # Update cut infeasibility
#             if m.log_level > 2
#                 inf_cut = -getvalue(vars[iSD, iSD]) * vecdot(vvj[no_i, no_i], getvalue(vars[no_i, no_i])) + (vecdot(vj[no_i], getvalue(vars[no_i, iSD])))^2
#                 if inf_cut > 0.
#                     spec_summ[:cut_max_n] += 1
#                     spec_summ[:cut_max] = max(inf_cut, spec_summ[:cut_max])
#                 elseif inf_cut < 0.
#                     spec_summ[:cut_min_n] += 1
#                     spec_summ[:cut_min] = max(-inf_cut, spec_summ[:cut_min])
#                 end
#             end
#         end
#     end
# end


#=========================================================
 Primal cuts functions
=========================================================#

# For each cone, calc outer inf and add if necessary, add primal cuts violated by current MIP solution
function calc_outer_inf_cuts!(m::PajaritoConicModel, add_viol_cuts::Bool, logs::Dict{Symbol,Real})
    tic()
    oa_viol = false
    cut_viol = false

    for n in 1:m.num_soc
        add_prim_cuts_soc!(m, add_viol_cuts, oa_viol, cut_viol, m.dim_soc[n], m.vars_soc[n], m.vars_dagg_soc[n], m.coefs_soc[n], m.summ_soc)
    end
    for n in 1:m.num_exp
        add_prim_cuts_exp!(m, add_viol_cuts, oa_viol, cut_viol, m.vars_exp[n], m.coefs_exp[n], m.summ_exp)
    end
    for n in 1:m.num_sdp
        add_prim_cuts_sdp!(m, add_viol_cuts, oa_viol, cut_viol, m.dim_sdp[n], m.vars_smat_sdp[n], m.coefs_smat_sdp[n], m.smat_sdp[n], m.summ_sdp)
    end

    print_inf_outer(m)
    logs[:outer_inf] += toq()

    return (oa_viol, cut_viol)
end

# Add primal cuts for a SOC
function add_prim_cuts_soc!(m::PajaritoConicModel, add_viol_cuts::Bool, oa_viol::Bool, cut_viol::Bool, dim::Int, vars::Vector{JuMP.Variable}, vars_dagg::Vector{JuMP.Variable}, coefs::Vector{Float64}, spec_summ::Dict{Symbol,Real})
    # Calculate and update outer infeasibility
    inf_outer = norm(coefs[2:end] .* getvalue(vars[2:end])) - (coefs[1] * getvalue(vars[1]))
    if m.log_level > 2
        update_inf_outer!(inf_outer, spec_summ)
    end

    # If outer infeasibility is small, return, else update and return if not adding primal cuts
    if inf_outer < m.primal_cut_inf_tol
        return
    end
    oa_viol = true
    if !add_viol_cuts
        return
    end

    # TODO are there other cases where we don't want to add this?
    # TODO should we process: clean zeros etc?
    if m.disagg_soc
        # Don't add primal cut if epigraph variable is zero
        # TODO can still add a different cut if infeasible
        if (coefs[1] * getvalue(vars[1])) < m.primal_cut_zero_tol
            return
        end

        for j in 2:dim
            # Add disagg primal cut (divide by original epigraph variable)
            # 2*dj >= 2xj`/y`*xj - (xj'/y`)^2*y
            @expression(m.model_mip, cut_expr, ((coefs[j] * getvalue(vars[j])) / (coefs[1] * getvalue(vars[1])))^2 * coefs[1] * vars[1] + 2. * vars_dagg[j-1] - (2 * (coefs[j] * getvalue(vars[j])) / (coefs[1] * getvalue(vars[1]))) * coefs[j] * vars[j])
            if getvalue(cut_expr) < m.primal_cut_zero_tol
                if m.mip_solver_drives
                    @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                else
                    @constraint(m.model_mip, cut_expr >= 0.)
                end
                cut_viol = true
            end
        end
    else
        # Don't add primal cut if norm of non-epigraph variables is zero
        # TODO can still add a different cut if infeasible
        solnorm = sqrt(sumabs2(coefs[2:dim] .* getvalue(vars[2:dim])))
        if solnorm < m.primal_cut_zero_tol
            return
        end

        # Add full primal cut
        # x`*x / ||x`|| <= y
        @expression(m.model_mip, cut_expr, coefs[1] * vars[1] - sum{(getvalue(vars[j]) * coefs[j]) / solnorm * (coefs[j] * vars[j]), j in 2:dim})
        if getvalue(cut_expr) < m.primal_cut_zero_tol
            if m.mip_solver_drives
                @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
            else
                @constraint(m.model_mip, cut_expr >= 0.)
            end
            cut_viol = true
        end
    end
end

# Add primal cut for a ExpPrimal cone
function add_prim_cuts_exp!(m::PajaritoConicModel, add_viol_cuts::Bool, oa_viol::Bool, cut_viol::Bool, vars::Vector{JuMP.Variable}, coefs::Vector{Float64}, spec_summ::Dict{Symbol,Real})
    inf_outer = coefs[2] * getvalue(vars[2]) * exp(coefs[1] * getvalue(vars[1]) / (coefs[2] * getvalue(vars[2]))) - coefs[3] * getvalue(vars[3])
    if m.log_level > 2
        update_inf_outer!(inf_outer, spec_summ)
    end

    # If outer infeasibility is small, return, else update and return if not adding primal cuts
    if inf_outer < m.primal_cut_inf_tol
        return
    end
    oa_viol = true
    if !add_viol_cuts
        return
    end

    # Don't add primal cut if perspective variable is zero
    # TODO can still add a different cut if infeasible
    if (coefs[2] * getvalue(vars[2])) < m.primal_cut_zero_tol
        return
    end

    # Add primal cut
    # y`e^(x`/y`) + e^(x`/y`)*(x-x`) + (e^(x`/y`)(y`-x`)/y`)*(y-y`) = e^(x`/y`)*(x + (y`-x`)/y`*y) = e^(x`/y`)*(x+(1-x`/y`)*y) <= z
    @expression(m.model_mip, cut_expr, coefs[3] * vars[3] - exp(coefs[1] * getvalue(vars[1]) / (coefs[2] * getvalue(vars[2]))) * (coefs[1] * vars[1] + (1. - (coefs[1] * getvalue(vars[1])) / (coefs[2] * getvalue(vars[2]))) * coefs[2] * vars[2]))
    if getvalue(cut_expr) < m.primal_cut_zero_tol
        if m.mip_solver_drives
            @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
        else
            @constraint(m.model_mip, cut_expr >= 0.)
        end
        cut_viol = true
    end
end

# Add primal cuts for a SDP cone
function add_prim_cuts_sdp!(m::PajaritoConicModel, add_viol_cuts::Bool, oa_viol::Bool, cut_viol::Bool, dim::Int, vars_smat::Array{JuMP.Variable,2}, coefs_smat::Array{Float64,2}, smat::Array{Float64,2}, spec_summ::Dict{Symbol,Real})
    # Convert solution to lower smat space and store in preallocated smat matrix
    for j in 1:dim, i in j:dim
        smat[i, j] = coefs_smat[i, j] * getvalue(vars_smat[i, j])
    end

    # Get eigendecomposition of smat solution (use symmetric property), save eigenvectors in smat matrix
    # TODO only need eigenvalues if not using primal cuts
    (eigvals, _) = LAPACK.syev!('V', 'L', smat)

    inf_outer = -minimum(eigvals)
    if m.log_level > 2
        update_inf_outer!(inf_outer, spec_summ)
    end

    # If outer infeasibility is small, return, else update and return if not adding primal cuts
    if inf_outer < m.primal_cut_inf_tol
        return
    end
    oa_viol = true
    if !add_viol_cuts
        return
    end

    # Add super-rank linear primal cut
    @expression(m.model_mip, cut_expr, sum{-eigvals[v] * smat[vi, v] * smat[vj, v] * (vi == vj ? 1. : 2.) * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim, v in 1:dim; eigvals[v] < 0.})
    if getvalue(cut_expr) < m.primal_cut_zero_tol
        if m.mip_solver_drives && m.oa_started
            @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
        else
            @constraint(m.model_mip, cut_expr >= 0.)
        end
        cut_viol = true
    end

    # TODO these help for iterative, but seem to slow down MIP-driven
    if m.sdp_eig
        for v in 1:dim
            if eigvals[v] < m.sdp_tol_eigval
                # 3 Add non-sparse rank-1 cut from smat eigenvector v
                @expression(m.model_mip, cut_expr, sum{(vi == vj ? 1. : 2.) * smat[vi, v] * smat[vj, v] * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim})
                if getvalue(cut_expr) < m.primal_cut_zero_tol
                    if m.mip_solver_drives
                        @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                    else
                        @constraint(m.model_mip, cut_expr >= 0.)
                    end
                    cut_viol = true
                end

                # These may slow down MIP
                # # 2 Sanitize eigenvector v for sparser rank-1 cut
                # # TODO try for multiple levels of sparsity
                # for vi in 1:dim
                #     if abs(smat[vi, v]) < m.sdp_tol_eigvec
                #         smat[vi, v] = 0.
                #     end
                # end
                #
                # # 3 Add sparse rank-1 svec cut from smat sparsified eigenvector v
                # @expression(m.model_mip, cut_expr, sum{(vi == vj ? 1. : 2.) * smat[vi, v] * smat[vj, v] * coefs_smat[vi, vj] * vars_smat[vi, vj], vj in 1:dim, vi in vj:dim})
                # if getvalue(cut_expr) < m.primal_cut_zero_tol
                #     if m.mip_solver_drives
                #         @lazyconstraint(m.cb_lazy, cut_expr >= 0.)
                #     else
                #         @constraint(m.model_mip, cut_expr >= 0.)
                #     end
                #     cut_viol = true
                # end
            end
        end
    end
end

# Update outer approximation infeasibility values in cone summary
function update_inf_outer!(inf_outer::Float64, spec_summ::Dict{Symbol,Real})
    if inf_outer > 0.
        spec_summ[:outer_max_n] += 1
        spec_summ[:outer_max] = max(inf_outer, spec_summ[:outer_max])
    elseif inf_outer < 0.
        spec_summ[:outer_min_n] += 1
        spec_summ[:outer_min] = max(-inf_outer, spec_summ[:outer_min])
    end
end


#=========================================================
 Algorithm utilities
=========================================================#

# Construct and warm-start MIP solution using best solution
function set_best_soln!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()
    if m.mip_solver_drives
        for ind in 1:length(m.cols_int)
            setsolutionvalue(m.cb_heur, m.x_mip[m.cols_int][ind], m.best_int[ind])
        end

        for ind in 1:length(m.cols_cont)
            setsolutionvalue(m.cb_heur, m.x_mip[m.cols_cont][ind], m.best_sub[ind])
        end

        for n in 1:m.num_soc
            for ind in 1:m.dim_soc[n]
                if m.isslacknew_soc[n][ind]
                    setsolutionvalue(m.cb_heur, m.vars_soc[n][ind], m.slck_sub[m.rows_sub_soc[n]][ind])
                end
            end

            if m.disagg_soc
                if m.slck_sub[m.rows_sub_soc[n]][1] == 0.
                    for ind in 2:m.dim_soc[n]
                        setsolutionvalue(m.cb_heur, m.vars_dagg_soc[n][ind-1], 0.)
                    end
                else
                    for ind in 2:m.dim_soc[n]
                        setsolutionvalue(m.cb_heur, m.vars_dagg_soc[n][ind-1], (m.slck_sub[m.rows_sub_soc[n]][ind]^2 / (2. * m.slck_sub[m.rows_sub_soc[n]][1])))
                    end
                end
            end
        end

        for n in 1:m.num_exp
            for ind in 1:3
                if m.isslacknew_exp[n][ind]
                    setsolutionvalue(m.cb_heur, m.vars_exp[n][ind], m.slck_sub[m.rows_sub_exp[n]][ind])
                end
            end
        end

        for n in 1:m.num_sdp
            for ind in 1:m.dim_sdp[n]
                if m.isslacknew_sdp[n][ind]
                    setsolutionvalue(m.cb_heur, m.vars_svec_sdp[n][ind], m.slck_sub[m.rows_sub_sdp[n]][ind])
                end
            end
        end
    else
        for ind in 1:length(m.cols_int)
            setvalue(m.x_mip[m.cols_int][ind], m.best_int[ind])
        end

        for ind in 1:length(m.cols_cont)
            setvalue(m.x_mip[m.cols_cont][ind], m.best_sub[ind])
        end

        for n in 1:m.num_soc
            for ind in 1:m.dim_soc[n]
                if m.isslacknew_soc[n][ind]
                    setvalue(m.vars_soc[n][ind], m.slck_sub[m.rows_sub_soc[n]][ind])
                end
            end

            if m.disagg_soc
                if m.slck_sub[m.rows_sub_soc[n]][1] == 0.
                    for ind in 2:m.dim_soc[n]
                        setvalue(m.vars_dagg_soc[n][ind-1], 0.)
                    end
                else
                    for ind in 2:m.dim_soc[n]
                        setvalue(m.vars_dagg_soc[n][ind-1], (m.slck_sub[m.rows_sub_soc[n]][ind]^2 / (2. * m.slck_sub[m.rows_sub_soc[n]][1])))
                    end
                end
            end
        end

        for n in 1:m.num_exp
            for ind in 1:3
                if m.isslacknew_exp[n][ind]
                    setvalue(m.vars_exp[n][ind], m.slck_sub[m.rows_sub_exp[n]][ind])
                end
            end
        end

        for n in 1:m.num_sdp
            for ind in 1:m.dim_sdp[n]
                if m.isslacknew_sdp[n][ind]
                    setvalue(m.vars_svec_sdp[n][ind], m.slck_sub[m.rows_sub_sdp[n]][ind])
                end
            end
        end
    end
    logs[:conic_soln] += toq()
    logs[:n_feas_add] += 1
end

# Reset all summary values for all cones in preparation for next iteration
function reset_cone_summary!(m::PajaritoConicModel)
    if m.log_level <= 2
        return
    end

    if m.num_soc > 0
        m.summ_soc[:outer_max_n] = 0
        m.summ_soc[:outer_max] = 0.
        m.summ_soc[:outer_min_n] = 0
        m.summ_soc[:outer_min] = 0.
        m.summ_soc[:dual_max_n] = 0
        m.summ_soc[:dual_max] = 0.
        m.summ_soc[:dual_min_n] = 0
        m.summ_soc[:dual_min] = 0.
        m.summ_soc[:cut_max_n] = 0
        m.summ_soc[:cut_max] = 0.
        m.summ_soc[:cut_min_n] = 0
        m.summ_soc[:cut_min] = 0.
    end

    if m.num_exp > 0
        m.summ_exp[:outer_max_n] = 0
        m.summ_exp[:outer_max] = 0.
        m.summ_exp[:outer_min_n] = 0
        m.summ_exp[:outer_min] = 0.
        m.summ_exp[:dual_max_n] = 0
        m.summ_exp[:dual_max] = 0.
        m.summ_exp[:dual_min_n] = 0
        m.summ_exp[:dual_min] = 0.
        m.summ_exp[:cut_max_n] = 0
        m.summ_exp[:cut_max] = 0.
        m.summ_exp[:cut_min_n] = 0
        m.summ_exp[:cut_min] = 0.
    end

    if m.num_sdp > 0
        m.summ_sdp[:outer_max_n] = 0
        m.summ_sdp[:outer_max] = 0.
        m.summ_sdp[:outer_min_n] = 0
        m.summ_sdp[:outer_min] = 0.
        m.summ_sdp[:dual_max_n] = 0
        m.summ_sdp[:dual_max] = 0.
        m.summ_sdp[:dual_min_n] = 0
        m.summ_sdp[:dual_min] = 0.
        m.summ_sdp[:cut_max_n] = 0
        m.summ_sdp[:cut_max] = 0.
        m.summ_sdp[:cut_min_n] = 0
        m.summ_sdp[:cut_min] = 0.
    end
end

# Transform svec vector into symmetric smat matrix
function make_smat!(svec::Vector{Float64}, smat::Array{Float64,2}, dim::Int)
    kSD = 1
    for jSD in 1:dim, iSD in jSD:dim
        if jSD == iSD
            smat[iSD, jSD] = svec[kSD]
        else
            smat[iSD, jSD] = smat[jSD, iSD] = svec[kSD] / sqrt(2)
        end
        kSD += 1
    end
    return smat
end


#=========================================================
 Logging, printing, testing functions
=========================================================#

# Create dictionary of logs for timing and iteration counts
function create_logs()
    logs = Dict{Symbol,Real}()

    # Timers
    logs[:total] = 0.       # Performing total optimize algorithm
    logs[:data_trans] = 0.  # Transforming data
    logs[:data_conic] = 0.  # Generating conic data
    logs[:data_mip] = 0.    # Generating MIP data
    logs[:relax_solve] = 0. # Solving initial conic relaxation model
    logs[:oa_alg] = 0.      # Performing outer approximation algorithm
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:conic_load] = 0.  # Loading conic subproblem model
    logs[:conic_solve] = 0. # Solving conic subproblem model
    logs[:conic_soln] = 0.  # Checking and adding new feasible conic solution
    logs[:dual_cuts] = 0.   # Adding dual cuts
    logs[:outer_inf] = 0.   # Calculating outer inf and adding primal cuts

    # Counters
    logs[:n_conic] = 0      # Number of conic subproblem solves
    logs[:n_feas_add] = 0   # Number of times feasible MIP solution is added
    logs[:n_repeat] = 0     # Number of times integer solution repeats

    return logs
end

# Print cone dimensions summary
function print_cones(m::PajaritoConicModel)
    if m.log_level == 0
        return
    end

    @printf "\n%-10s | %-8s | %-8s | %-8s\n" "Species" "Count" "Min dim" "Max dim"
    if m.num_soc > 0
        @printf "%10s | %8d | %8d | %8d\n" "SOC" m.num_soc m.summ_soc[:min_dim] m.summ_soc[:max_dim]
    end
    if m.num_exp > 0
        @printf "%10s | %8d | %8d | %8d\n" "ExpPrimal" m.num_exp 3 3
    end
    if m.num_sdp > 0
        @printf "%10s | %8d | %8d | %8d\n" "SDP" m.num_sdp m.summ_sdp[:min_dim] m.summ_sdp[:max_dim]
    end
    @printf "\n"
    flush(STDOUT)
end

# Print dual cone infeasibilities of dual vectors only
function print_inf_dual(m::PajaritoConicModel)
    if m.log_level <= 2
        return
    end

    @printf "\n%-10s | %-32s\n" "Species" "Dual cone infeas"
    @printf "%-10s | %-6s %-8s  %-6s %-8s\n" "" "Inf" "Worst" "Feas" "Worst"
    if m.num_soc > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "SOC" m.summ_soc[:dual_max_n] m.summ_soc[:dual_max] m.summ_soc[:dual_min_n] m.summ_soc[:dual_min]
    end
    if m.num_exp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "ExpPrimal" m.summ_exp[:dual_max_n] m.summ_exp[:dual_max] m.summ_exp[:dual_min_n] m.summ_exp[:dual_min]
    end
    if m.num_sdp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "SDP" m.summ_sdp[:dual_max_n] m.summ_sdp[:dual_max] m.summ_sdp[:dual_min_n] m.summ_sdp[:dual_min]
    end
    @printf "\n"
    flush(STDOUT)
end

# Print infeasibilities of dual vectors and dual cuts added to MIP
function print_inf_dualcuts(m::PajaritoConicModel)
    if m.log_level <= 2
        return
    end

    @printf "\n%-10s | %-32s | %-32s\n" "Species" "Dual cone infeas" "Cut infeas"
    @printf "%-10s | %-6s %-8s  %-6s %-8s | %-6s %-8s  %-6s %-8s\n" "" "Inf" "Worst" "Feas" "Worst" "Inf" "Worst" "Feas" "Worst"
    if m.num_soc > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e\n" "SOC" m.summ_soc[:dual_max_n] m.summ_soc[:dual_max] m.summ_soc[:dual_min_n] m.summ_soc[:dual_min] m.summ_soc[:cut_max_n] m.summ_soc[:cut_max] m.summ_soc[:cut_min_n] m.summ_soc[:cut_min]
    end
    if m.num_exp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e\n" "ExpPrimal" m.summ_exp[:dual_max_n] m.summ_exp[:dual_max] m.summ_exp[:dual_min_n] m.summ_exp[:dual_min] m.summ_exp[:cut_max_n] m.summ_exp[:cut_max] m.summ_exp[:cut_min_n] m.summ_exp[:cut_min]
    end
    if m.num_sdp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e\n" "SDP" m.summ_sdp[:dual_max_n] m.summ_sdp[:dual_max] m.summ_sdp[:dual_min_n] m.summ_sdp[:dual_min] m.summ_sdp[:cut_max_n] m.summ_sdp[:cut_max] m.summ_sdp[:cut_min_n] m.summ_sdp[:cut_min]
    end
    @printf "\n"
    flush(STDOUT)
end

# Print outer approximation infeasibilities of MIP solution
function print_inf_outer(m::PajaritoConicModel)
    if m.log_level <= 2
        return
    end

    @printf "\n%-10s | %-32s\n" "Species" "Outer approx infeas"
    @printf "%-10s | %-6s %-8s  %-6s %-8s\n" "" "Inf" "Worst" "Feas" "Worst"
    if m.num_soc > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "SOC" m.summ_soc[:outer_max_n] m.summ_soc[:outer_max] m.summ_soc[:outer_min_n] m.summ_soc[:outer_min]
    end
    if m.num_exp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "ExpPrimal" m.summ_exp[:outer_max_n] m.summ_exp[:outer_max] m.summ_exp[:outer_min_n] m.summ_exp[:outer_min]
    end
    if m.num_sdp > 0
        @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" "SDP" m.summ_sdp[:outer_max_n] m.summ_sdp[:outer_max] m.summ_sdp[:outer_min_n] m.summ_sdp[:outer_min]
    end
    @printf "\n"
    flush(STDOUT)
end

# Print objective gap information
function print_gap(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level <= 1
        return
    end

    if (logs[:n_conic] == 1) || (m.log_level > 2)
        @printf "\n%-4s | %-14s | %-14s | %-11s | %-11s\n" "Iter" "Best obj" "OA obj" "Rel gap" "Time (s)"
    end
    if m.gap_rel_opt < 1000
        @printf "%4d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" logs[:n_conic] m.best_obj m.mip_obj m.gap_rel_opt (time() - logs[:oa_alg])
    elseif isnan(m.gap_rel_opt)
        @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" logs[:n_conic] m.best_obj m.mip_obj "Inf" (time() - logs[:oa_alg])
    else
        @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" logs[:n_conic] m.best_obj m.mip_obj ">1000" (time() - logs[:oa_alg])
    end
    flush(STDOUT)
end

# Print after finish
function print_finish(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level == 0
        return
    end

    if m.mip_solver_drives
        @printf "\nFinished MIP-solver-driven outer approximation algorithm:\n"
    else
        @printf "\nFinished iterative outer approximation algorithm:\n"
    end
    @printf " - Total time (s)       = %14.2e\n" logs[:total]
    @printf " - Status               = %14s\n" m.status
    @printf " - Best feasible obj.   = %+14.6e\n" m.best_obj
    @printf " - Final OA obj. bound  = %+14.6e\n" m.mip_obj
    @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt
    @printf " - Conic iter. count    = %14d\n" logs[:n_conic]
    @printf " - Feas. solution count = %14d\n" logs[:n_feas_add]
    @printf " - Integer repeat count = %14d\n" logs[:n_repeat]
    @printf "\nTimers (s):\n"
    @printf " - Setup                = %14.2e\n" (logs[:total] - logs[:oa_alg])
    @printf " -- Transform data      = %14.2e\n" logs[:data_trans]
    @printf " -- Create conic data   = %14.2e\n" logs[:data_conic]
    @printf " -- Create MIP data     = %14.2e\n" logs[:data_mip]
    @printf " -- Load/solve relax    = %14.2e\n" logs[:relax_solve]
    if m.mip_solver_drives
        @printf " - MIP-driven algorithm = %14.2e\n" logs[:oa_alg]
    else
        @printf " - Iterative algorithm  = %14.2e\n" logs[:oa_alg]
        @printf " -- Solve MIPs          = %14.2e\n" logs[:mip_solve]
    end
    @printf " -- Load conic data     = %14.2e\n" logs[:conic_load]
    @printf " -- Solve conic model   = %14.2e\n" logs[:conic_solve]
    @printf " -- Use conic solution  = %14.2e\n" logs[:conic_soln]
    @printf " -- Add dual cuts       = %14.2e\n" logs[:dual_cuts]
    @printf " -- Use outer inf/cuts  = %14.2e\n" logs[:outer_inf]
    @printf "\n"
    flush(STDOUT)

    if m.log_level < 2
        return
    end
end
