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
- some infeasible solutions from heuristic callback? maybe want to project the primal solutions onto primal cones (can reuse code from dual projections)
- maybe give MIP solvers command to run gap to 0 (otherwise may cycle if not goes to 0)
- stop 0 = 0 constraints being added
- MPB issue - can't call supportedcones on defaultConicsolver

TODO features
- want to be able to query logs information etc
- use JP updated SOC disagg_soc with half as many cuts
- for SOCRotated disagg_soc cuts
-- maybe add two disagg_soc cuts - one for dividing by p and one for dividing by q
-- or reformulate pq >= sum x^2 using 3-dim SOCRotated pq >= w^2 and n dim SOC w^2 >= sum x^2
- have option to only add violated cuts (especially for SDP, where each SOC cut slows down mip and we have many SOC cuts)
- print cone info to one file and gap info to another file
- maybe if experience conic problem strong duality failure, use a no-good cut on that integer solution and proceed. but that could cut off optimal sol?
- dual cone projection - implement multiple heuristic projections and optimal euclidean projection
- determine whether projecting from cone interior helps
- integers in nonlinear var cones
-- add cuts directly on integer vars (don't duplicate like the old way - otherwise this is equivalent)
-- only variables removed from conic should be integer linear cone ones
-- need extra constraint zero cones and extend b_conic so can effectively fix values of variables in conic that are integer and in nonlinear cones
- currently all SDP sanitized eigvecs have norm 1, but may want to multiply V by say 100 (or perhaps largest eigenvalue) before removing zeros, to get more significant digits
- could automatically give timeout and gap tol as options to mip solver for mip-driven. and for iterative, probably want timeout too in case MIP solves are extremely slow
- when JuMP can handle anonymous variables, use that syntax
- could check that primal solution is feasible for the original problem. if the solver is well-behaved then x_i^2/x_0 should still tend to zero if the x_i^2 and x_0 are approximately zero
=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
    log_level::Int              # Verbosity flag: 1 for minimal OA iteration and solve statistics, 2 for including cone summary information, 3 for running commentary
    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    soc_in_mip::Bool            # Use SOC/SOCRotated cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    disagg_soc::Bool            # Disaggregate SOC/SOCRotated cones in the MIP only (if solver is conic)
    drop_dual_infeas::Bool      # Do not add cuts from dual cone infeasible dual vectors
    proj_dual_infeas::Bool      # Project dual cone infeasible dual vectors onto dual cone boundaries
    proj_dual_feas::Bool        # Project dual cone strictly feasible dual vectors onto dual cone boundaries
    mip_solver::MathProgBase.AbstractMathProgSolver # MIP solver
    cont_solver::MathProgBase.AbstractMathProgSolver # Continuous solver
    timeout::Float64            # Time limit for outer approximation algorithm not including initial load (in seconds)
    rel_gap::Float64            # Relative optimality gap termination condition
    zero_tol::Float64           # Tolerance for setting small absolute values to zeros
    sdp_init_soc::Bool          # Use SDP initial SOC cuts (if MIP solver supports MISOCP)
    sdp_eig::Bool               # Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    sdp_tol_eigvec::Float64     # Tolerance for setting small values in SDP eigenvectors to zeros (for cut sanitation)
    sdp_tol_eigval::Float64     # Tolerance for ignoring eigenvectors corresponding to small (positive) eigenvalues

    # Internal switches
    _soc_in_mip::Bool           # Only if using MIP solver supporting MISOCP
    _sdp_init_soc::Bool         # Only if using MIP solver supporting MISOCP
    _sdp_soc::Bool              # Only if using MIP solver supporting MISOCP (cannot add SOCs during MIP-driven solve)

    # Initial conic data
    num_var_orig::Int           # Initial number of variables
    num_con_orig::Int           # Initial number of constraints
    c_orig::Vector{Float64}     # Initial objective coefficients vector
    A_orig::SparseMatrixCSC{Float64,Int64} # Initial affine constraint matrix (sparse representation)
    b_orig::Vector{Float64}     # Initial constraint right hand side
    cone_con_orig::Vector{Tuple{Symbol,Vector{Int}}} # Initial constraint cones vector (cone, index)
    cone_var_orig::Vector{Tuple{Symbol,Vector{Int}}} # Initial variable cones vector (cone, index)

    # Mutable variable data
    types_orig::Vector{Symbol}  # Variable types vector on original variables (only :Bin, :Cont, :Int)
    start_orig::Vector{Float64} # Variable warm start vector on original variables

    # Conic constructed data
    cols_bint::Vector{Int}      # Indices of integer variables in original data
    cols_cont::Vector{Int}      # Indices of all continuous variables in original data
    cols_linr::Vector{Int}      # Indices of linear continuous variables in original data
    oldnew_col::Vector{Int}     # Map for original column indices to new conic column indices
    oldnew_row::Vector{Int}     # Map for original row indices to new conic row indices
    cone_con_full::Vector{Tuple{Symbol,Vector{Int}}} # Transformed constraint cone data for conic subproblem
    cone_var_cont::Vector{Tuple{Symbol,Vector{Int}}} # Transformed variable cone data for conic subproblem
    c_bint::Vector{Float64}     # Objective coefficient subvector for integer variables
    c_cont::Vector{Float64}     # Objective coefficient subvector for continuous variables
    A_bint::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and integer variable columns
    A_cont::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    b_full::Vector{Float64}     # Subvector of b containing full rows

    # MIP constructed data
    model_mip::JuMP.Model       # JuMP MIP (outer approximation) model
    x_bint::Vector{JuMP.Variable} # JuMP integer variables vector
    x_all::Vector{JuMP.Variable} # JuMP vector of all variables for heuristics and warm-starting
    num_cone_nlnr::Int          # Count of nonlinear cones
    map_ifvar::Vector{Bool}     # Bool for whether nonlinear cone is a variable cone in the conic problems
    map_spec::Vector{Symbol}    # Species of nonlinear cone
    map_ind::Vector{Vector{Int}} # Indices (rows or columns) of cone in original (conic relaxation) data
    map_ind_new::Vector{Vector{Int}} # Indices (rows or columns) of cone in transformed (conic subproblem) data
    map_Asub::Vector{SparseMatrixCSC{Float64,Int64}} # Submatrix of A_cont containing rows corresponding to nonlinear constraint cone
    map_vars::Vector{Vector{JuMP.Variable}} # Variables associated with nonlinear cone
    map_vars_dagg::Vector{Vector{JuMP.Variable}} # Disaggregated variables associated with nonlinear cone
    map_vars_help::Vector{Vector{JuMP.Variable}} # Helper variables associated with nonlinear cone
    map_vars_smat::Vector{Array{JuMP.Variable,2}} # SDP original and helper variables in smat form

    # Dynamic solve data
    summary::Dict{Symbol,Dict{Symbol,Real}} # Infeasibilities (outer, cut, dual) of each cone species at current iteration
    bc_started::Bool            # Bool for whether MIP-driven solve has begun
    status::Symbol              # Current solve status
    obj_mip::Float64            # Latest MIP (outer approx) objective value
    obj_best::Float64           # Best conic (feasible) objective value
    gap_rel_opt::Float64        # Relative optimality gap = |obj_mip - obj_best|/|obj_best|
    soln_best::Vector{Float64}  # Best original solution vector (corresponding to best objective)
    queue_heur::Vector{Vector{Float64}} # Heuristic queue for x_all

    # Model constructor
    function PajaritoConicModel(log_level, mip_solver_drives, soc_in_mip, disagg_soc, drop_dual_infeas, proj_dual_infeas, proj_dual_feas, mip_solver, cont_solver, timeout, rel_gap, zero_tol, sdp_init_soc, sdp_eig, sdp_soc, sdp_tol_eigvec, sdp_tol_eigval)
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.soc_in_mip = soc_in_mip
        m.disagg_soc = disagg_soc
        m.drop_dual_infeas = drop_dual_infeas
        m.proj_dual_infeas = proj_dual_infeas
        m.proj_dual_feas = proj_dual_feas
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.timeout = timeout
        m.rel_gap = rel_gap
        m.zero_tol = zero_tol
        m.sdp_init_soc = sdp_init_soc
        m.sdp_eig = sdp_eig
        m.sdp_soc = sdp_soc
        m.sdp_tol_eigvec = sdp_tol_eigvec
        m.sdp_tol_eigval = sdp_tol_eigval

        @printf "\n\n"

        # Determine whether to use MISOCP outer approximation MIP
        if m.soc_in_mip
            mip_species = MathProgBase.supportedcones(m.mip_solver)
            if ((:SOC in mip_species) && (:SOCRotated in mip_species))
                m._soc_in_mip = true
            else
                warn("MIP solver specified does not support MISOCP; defaulting to MILP outer approximation model\n")
                m._soc_in_mip = false
            end
        else
            m._soc_in_mip = false
        end

        # Determine which SOC cuts to use for SDPs
        if m._soc_in_mip
            m._sdp_init_soc = m.sdp_init_soc
            if m.sdp_soc
                if m.mip_solver_drives
                    warn("Rotated-SOC cuts for SDP cones cannot be added during the MIP-solver-driven algorithm, but will be used for initial cuts\n")
                end
                m._sdp_soc = true
            end
        else
            m._sdp_init_soc = false
            m._sdp_soc = false
        end

        # TODO automatically pass as options to MIP solver
        if m.mip_solver_drives
            warn("For the MIP-solver-driven algorithm, time limit and optimality tolerance must be specified as MIP solver options, not Pajarito options\n")
        end

        if m.drop_dual_infeas && m.proj_dual_infeas
            warn("Cannot both drop and project dual cone infeasible cuts; defaulting to keeping and projecting\n")
        end

        # Initialize data
        m.types_orig = Symbol[]
        m.start_orig = Float64[]
        m.bc_started = false
        m.num_var_orig = 0
        m.num_con_orig = 0
        m.obj_best = Inf
        m.soln_best = Float64[]
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
    for (species, inds) in cone_con
        if species == :Free
            error("A cone $species is in the constraint cones\n")
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
    for (species, inds) in cone_var
        if species == :Zero
            error("A cone $species is in the variable cones\n")
        end

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
        conic_species = MathProgBase.supportedcones(m.cont_solver)
        for (species, inds) in vcat(cone_con, cone_var)
            if !(species in conic_species)
                error("Cones $species are not supported by the specified conic solver\n")
            end
        end
    end

    # Verify consistency of cone indices and create cone summary dictionary with min/max dimensions of each species
    summary = Dict{Symbol,Dict{Symbol,Real}}()
    for (species, inds) in vcat(cone_con, cone_var)
        # Verify dimensions of cones
        if isempty(inds)
            error("A cone $species has no associated indices\n")
        end
        if species == :SOC && (length(inds) < 2)
            error("A cone $species has fewer than 2 indices ($(length(inds)))\n")
        elseif species == :SOCRotated && (length(inds) < 3)
            error("A cone $species has fewer than 3 indices ($(length(inds)))\n")
        elseif species == :SDP
            if length(inds) < 3
                error("A cone $species has fewer than 3 indices ($(length(inds)))\n")
            else
                if floor(sqrt(8 * length(inds) + 1)) != sqrt(8 * length(inds) + 1)
                    error("A cone $species (in SD svec form) does not have a valid (triangular) number of indices ($(length(inds)))\n")
                end
            end
        elseif species == :ExpPrimal && (length(inds) != 3)
            error("A cone $species does not have exactly 3 indices ($(length(inds)))\n")
        end

        # Create cone summary dictionary
        if species in (:SOC, :SOCRotated, :SDP, :ExpPrimal)
            if !haskey(summary, species)
                summary[species] = Dict{Symbol,Real}(:count => 1, :max_dim => length(inds), :min_dim => length(inds))
            else
                summary[species][:count] += 1
                if (summary[species][:max_dim] < length(inds))
                    summary[species][:max_dim] = length(inds)
                elseif (summary[species][:min_dim] > length(inds))
                    summary[species][:min_dim] = length(inds)
                end
            end
        end
    end

    # Print cones summary
    print_cones(m, summary)

    # Check for values in A smaller than zero tolerance
    A_sp = sparse(A)
    A_num_zeros = count(val -> (abs(val) < m.zero_tol), nonzeros(A_sp))
    if A_num_zeros > 0
        @printf "Matrix A has %d entries smaller than zero tolerance %e; performance may be improved by first fixing small magnitudes to zero\n" A_num_zeros m.zero_tol
    end

    m.num_con_orig = num_con_orig
    m.num_var_orig = num_var_orig
    m.A_orig = A_sp
    m.c_orig = c
    m.b_orig = b
    m.cone_con_orig = Tuple{Symbol,Vector{Int}}[(species, collect(inds)) for (species, inds) in cone_con]
    m.cone_var_orig = Tuple{Symbol,Vector{Int}}[(species, collect(inds)) for (species, inds) in cone_var]
    m.summary = summary
    m.soln_best = fill(NaN, m.num_var_orig)
    m.status = :Loaded

    # @printf "\n\n\n"
    # (I,J,V) = findnz(A_sp)
    # @show I
    # @show J
    # @show V
    # @show c
    # @show b
    # @show cone_con
    # @show cone_var
end

# Store warm-start vector on original variables in Pajarito model
function MathProgBase.setwarmstart!(m::PajaritoConicModel, start_orig::Vector{Real})
    # Check if vector can be loaded
    if m.status != :Loaded
        error("Must specify warm start right after loading problem\n")
    end
    if length(start_orig) != m.num_var_orig
        error("Warm start vector length ($(length(start_orig))) does not match number of variables ($(m.num_var_orig))\n")
    end

    m.start_orig = start_orig
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

    m.types_orig = types_var
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    # Initialize
    if m.status != :Loaded
        error("Must call optimize! function after loading problem\n")
    end
    if isempty(m.types_orig)
        error("Variable types were not specified; must call setvartype! function\n")
    end
    logs = create_logs()
    reset_cone_summary!(m)

    # Generate model data and instantiate MIP model
    logs[:total_setup] = time()
    create_conic_data!(m, logs)
    create_mip_data!(m, logs)

    # Solve relaxed conic problem, if feasible, add initial cuts to MIP model and use algorithm
    if process_relax!(m, logs)
        print_inf_dual(m)
        logs[:total_setup] = time() - logs[:total_setup]

        # Solve the transformed model with specified algorithm
        logs[:oa_alg] = time()
        if m.mip_solver_drives
            # MIP solver driven outer approximation algorithm
            solve_mip_driven!(m, logs)
        else
            # Iterative outer approximation algorithm
            solve_iterative!(m, logs)
        end
        logs[:oa_alg] = time() - logs[:oa_alg]
    else
        logs[:total_setup] = time() - logs[:total_setup]
    end

    # Print summary
    print_finish(m, logs)
end

# Redefine MathProgBase functions for getting model information
MathProgBase.numconstr(m::PajaritoConicModel) = m.num_con_orig
MathProgBase.numvar(m::PajaritoConicModel) = m.num_var_orig
MathProgBase.status(m::PajaritoConicModel) = m.status
MathProgBase.getobjval(m::PajaritoConicModel) = m.obj_best
MathProgBase.getsolution(m::PajaritoConicModel) = m.soln_best


#=========================================================
 Model constructor functions
=========================================================#

# Transfrom data for the conic relaxation model and subproblem models
function create_conic_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Build variable cone list for conic subproblem, variable map, and variable type lists
    num_cont = 0
    cols_bint = Int[]
    cols_cont = Int[]
    cols_linr = Int[]
    oldnew_col = zeros(Int, m.num_var_orig)
    cone_var_cont = Tuple{Symbol,Vector{Int}}[]

    for (species, cols) in m.cone_var_orig
        cols_cont_new = Int[]
        for j in cols
            if m.types_orig[j] == :Cont
                if species in (:Free, :NonNeg, :NonPos)
                    push!(cols_linr, j)
                end
                push!(cols_cont, j)
                num_cont += 1
                oldnew_col[j] = num_cont
                push!(cols_cont_new, num_cont)
            else
                if !in(species, (:Free, :NonNeg, :NonPos))
                    error("A variable of type $(m.types_orig[j]) is in a cone $species; integer variables in nonlinear cones is not currently supported (but we know how to implement this, so please open an issue and include your model data)\n")
                end
                push!(cols_bint, j)
            end
        end
        if !isempty(cols_cont_new)
            push!(cone_var_cont, (species, cols_cont_new))
        end
    end

    # Determine which constraints are empty of continuous variables
    (A_cont_I, _, _) = findnz(m.A_orig[:, cols_cont])
    count_row_nz = zeros(Int, m.num_con_orig)
    for row in A_cont_I
        count_row_nz[row] += 1
    end

    # Build constraint cone list, constraint map, and list of constraints with continuous variables
    num_full = 0
    rows_full = Int[]
    oldnew_row = zeros(Int, m.num_con_orig)
    cone_con_full = Tuple{Symbol,Vector{Int}}[]

    for (species, rows) in m.cone_con_orig
        if species in (:Zero, :NonNeg, :NonPos)
            rows_full_new = Int[]
            for i in rows
                if count_row_nz[i] > 0
                    push!(rows_full, i)
                    num_full += 1
                    oldnew_row[i] = num_full
                    push!(rows_full_new, num_full)
                end
            end
            if !isempty(rows_full_new)
                push!(cone_con_full, (species, rows_full_new))
            end
        else
            push!(cone_con_full, (species, collect((num_full + 1):(num_full + length(rows)))))
            append!(rows_full, rows)
            oldnew_row[rows] = collect((num_full + 1):(num_full + length(rows)))
            num_full += length(rows)
        end
    end

    # Store conic data
    m.cols_bint = cols_bint
    m.cols_cont = cols_cont
    m.cols_linr = cols_linr
    m.oldnew_col = oldnew_col
    m.oldnew_row = oldnew_row
    m.cone_var_cont = cone_var_cont
    m.cone_con_full = cone_con_full
    m.A_bint = m.A_orig[rows_full, cols_bint]
    m.A_cont = m.A_orig[rows_full, cols_cont]
    m.c_bint = m.c_orig[cols_bint]
    m.c_cont = m.c_orig[cols_cont]
    m.b_full = m.b_orig[rows_full]

    # @show rows_full
    # @show cols_bint
    # @show cols_linr
    # @show cols_nlnr
    # @show cone_var_cont
    # @show cone_con_full
    # @show m.A_bint
    # @show m.A_cont
    # @show m.c_bint
    # @show m.c_cont
    # @show m.b_full

    logs[:conic_data] += toq()
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    # MIP data is
    # Min          c_orig*x_orig
    # st. b_orig - A_orig*x_orig - x_slck                   in Zero
    #     b_orig - A_orig*x_orig                            in linear(cone_con_orig)
    #                     x_orig                            in bounds(cone_var_orig)
    #                              x_slck                   in bounds(nonlinear(cone_con_orig))
    #              B_orig*x_orig          - B_dagg*x_dagg   in NonNeg
    tic()

    # Define model-building function for nonlinear cones
    function build_cone!(model_mip::JuMP.Model, vars::Vector{JuMP.Variable}, species::Symbol, num_cone_nlnr::Int, num_cone_dagg::Int, v_or_c::Char)
        # Initialize possibly empty disaggregated and helper variable vectors
        x_dagg_cone = JuMP.Variable[]
        x_help_cone = JuMP.Variable[]
        x_smat_cone = Array{JuMP.Variable,2}(0, 0)

        # Set start value for nonlinear cone to be zero vector, so can calculate initial cut infeasibilities
        for var in vars
            setvalue(var, 0.)
        end

        # Set bounds and print names, for SOCs disaggregate or add to MIP, for SDPs add initial SOC constraints
        if species == :SOC
            for (ind, var) in enumerate(vars)
                setname(var, "$(v_or_c)$(num_cone_nlnr)SOC$(ind)")
            end
            setlowerbound(vars[1], 0.)

            if m._soc_in_mip
                @constraint(model_mip, sum{v^2, v in vars[2:end]} <= vars[1]^2)
            elseif m.disagg_soc && (length(vars) > 2)
                num_cone_dagg += 1
                x_dagg_cone = @variable(model_mip, _[j in 1:(length(vars) - 1)] >= 0., basename="d$(num_cone_dagg)SOC", start=0.)
                @constraint(model_mip, vars[1] - sum(x_dagg_cone) >= 0.)
            end

        elseif species == :SOCRotated
            for (ind, var) in enumerate(vars)
                setname(var, "$(v_or_c)$(num_cone_nlnr)SOCR$(ind)")
            end
            setlowerbound(vars[1], 0.)
            setlowerbound(vars[2], 0.)

            if m._soc_in_mip
                x_help_cone = @variable(model_mip, _[1:1] >= 0., basename="h$(num_cone_nlnr)SOCRh", start=0.)
                @constraint(model_mip, x_help_cone[1] == 2 * vars[2])
                @constraint(model_mip, sum{v^2, v in vars[3:end]} <= vars[1] * x_help_cone[1])
            elseif m.disagg_soc && (length(vars) > 3)
                # TODO this only adds one cut for dividing by vars[2], but may need two. otherwise, add extra SOCRotated in 3 dim to conic and use SOC disagg_soc
                warn("Disaggregation for cones $species is not currently supported (but we know how to implement this, so please open an issue and include your model data)\n")
                # num_cone_dagg += 1
                # x_dagg_cone = @variable(model_mip, _[j in 1:(length(vars) - 2)] >= 0., basename="d$(num_cone_dagg)SOCR", start=0.)
                # @constraint(model_mip, 2 * vars[1] - sum(x_dagg_cone) >= 0.)
            end

        elseif species == :ExpPrimal
            for (ind, var) in enumerate(vars)
                setname(var, "$(v_or_c)$(num_cone_nlnr)ExPr$(ind)")
            end
            setlowerbound(vars[2], 0.)
            setlowerbound(vars[3], 0.)

        elseif species == :SDP
            # Set up svec space variable vector
            nSD = round(Int, sqrt(1/4 + 2 * length(vars)) - 1/2)
            kSD = 1
            for jSD in 1:nSD, iSD in jSD:nSD
                if jSD == iSD
                    setname(vars[kSD], "$(v_or_c)$(num_cone_nlnr)SDon$(kSD)")
                    setlowerbound(vars[kSD], 0.)
                else
                    setname(vars[kSD], "$(v_or_c)$(num_cone_nlnr)SDoff$(kSD)")
                end
                kSD += 1
            end

            if m._sdp_init_soc || m._sdp_soc
                # Set up smat space variable array and add optional SDP initial SOC and dynamic SOC cuts
                x_help_cone = @variable(model_mip, _[j in 1:nSD] >= 0, basename="h$(num_cone_nlnr)SDhelp", start=0.)
                x_smat_cone = Array{JuMP.Variable,2}(nSD, nSD)
                kSD = 1
                for jSD in 1:nSD, iSD in jSD:nSD
                    if jSD == iSD
                        # Create helper variable corresponding to diagonal element multiplied by sqrt(2)
                        @constraint(model_mip, x_help_cone[jSD] == vars[kSD] * sqrt(2))
                        x_smat_cone[jSD, iSD] = x_help_cone[jSD]
                    else
                        x_smat_cone[jSD, iSD] = x_smat_cone[iSD, jSD] = vars[kSD]
                        if m._sdp_init_soc
                            # Add initial rotated SOC for off-diagonal element to enforce 2x2 principal submatrix PSDness
                            @constraint(model_mip, x_help_cone[jSD] * x_help_cone[iSD] >= vars[kSD]^2)
                        end
                    end
                    kSD += 1
                end
            end
        end

        return (num_cone_dagg, x_dagg_cone, x_help_cone, x_smat_cone)
    end

    # Initialize JuMP model for MIP outer approximation problem
    model_mip = JuMP.Model(solver = m.mip_solver)

    # Create initial variables and set types and warm-start and objective
    x_orig = @variable(model_mip, _[1:m.num_var_orig])

    for j in 1:m.num_var_orig
        setcategory(x_orig[j], m.types_orig[j])
    end

    if !isempty(m.start_orig)
        for j in 1:m.num_var_orig
            setvalue(x_orig[j], m.start_orig[j])
        end
    end

    # Start to build vector of all variables, starting with integer then linear continuous variables then for each cone, the cone, disaggregated, and helper variables
    x_all = vcat(x_orig[m.cols_bint], x_orig[m.cols_linr])

    # Set objective function
    @objective(model_mip, :Min, dot(m.c_orig, x_orig))

    # Initialize maps for nonlinear cones
    num_cone_nlnr = 0
    num_cone_dagg = 0
    map_ifvar = Bool[]
    map_spec = Symbol[]
    map_ind = Vector{Int}[]
    map_ind_new = Vector{Int}[]
    map_Asub = SparseMatrixCSC{Float64,Int64}[]
    map_vars = Vector{JuMP.Variable}[]
    map_vars_dagg = Vector{JuMP.Variable}[]
    map_vars_help = Vector{JuMP.Variable}[]
    map_vars_smat = Vector{Array{JuMP.Variable,2}}(0)

    # For variable cones, add to MIP and store maps
    for (species, cols) in m.cone_var_orig
        if species in (:NonNeg, :NonPos, :Free)
            if species == :NonNeg
                for j in cols
                    setname(x_orig[j], "v$(j)")
                    setlowerbound(x_orig[j], 0.)
                end

            elseif species == :NonPos
                for j in cols
                    setname(x_orig[j], "v$(j)")
                    setupperbound(x_orig[j], 0.)
                end

            elseif species == :Free
                for j in cols
                    setname(x_orig[j], "v$(j)")
                end
            end

        elseif species in (:SOC, :SOCRotated, :ExpPrimal, :SDP)
            # Set nonlinear cone variables and add helper variables or disaggregate
            num_cone_nlnr += 1
            (num_cone_dagg, x_dagg_cone, x_help_cone, x_smat_cone) = build_cone!(model_mip, x_orig[cols], species, num_cone_nlnr, num_cone_dagg, 'v')

            # Put nonlinear cone data and variables into maps
            push!(map_ifvar, true)
            push!(map_spec, species)
            push!(map_ind, cols)
            push!(map_ind_new, m.oldnew_col[cols])
            push!(map_Asub, spzeros(0, 0))
            push!(map_vars, x_orig[cols])
            push!(map_vars_dagg, x_dagg_cone)
            push!(map_vars_help, x_help_cone)
            push!(map_vars_smat, x_smat_cone)

            append!(x_all, vcat(x_orig[cols], x_dagg_cone, x_help_cone))
        end
    end

    # For constraint cones, if linear, add constraints directly to MIP, else create slack variables and add constraints and pretend slacks are in variable cone, store maps
    lhs_expr = m.b_orig - m.A_orig*x_orig
    for (species, rows) in m.cone_con_orig
        if species == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0.)

        elseif species == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)

        elseif species == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)

        elseif species in (:SOC, :SOCRotated, :ExpPrimal, :SDP)
            # Create slacks and add slack equality constraint for nonlinear cone
            x_slck_cone = @variable(model_mip, _[1:length(rows)])
            @constraint(model_mip, lhs_expr[rows] - x_slck_cone .== 0.)

            # Set nonlinear cone variables and add helper variables or disaggregate
            num_cone_nlnr += 1
            (num_cone_dagg, x_dagg_cone, x_help_cone, x_smat_cone) = build_cone!(model_mip, x_slck_cone, species, num_cone_nlnr, num_cone_dagg, 'c')

            # Put nonlinear cone data and variables into maps
            push!(map_ifvar, false)
            push!(map_spec, species)
            push!(map_ind, rows)
            push!(map_ind_new, m.oldnew_row[rows])
            push!(map_Asub, sparse(m.A_cont[m.oldnew_row[rows], :]))
            push!(map_vars, x_slck_cone)
            push!(map_vars_dagg, x_dagg_cone)
            push!(map_vars_help, x_help_cone)
            push!(map_vars_smat, x_smat_cone)

            append!(x_all, vcat(x_slck_cone, x_dagg_cone, x_help_cone))
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_bint = x_orig[m.cols_bint]
    m.x_all = x_all

    # Store maps and conic info
    m.num_cone_nlnr = num_cone_nlnr
    m.map_ifvar = map_ifvar
    m.map_spec = map_spec
    m.map_ind = map_ind
    m.map_ind_new = map_ind_new
    m.map_Asub = map_Asub
    m.map_vars = map_vars
    m.map_vars_dagg = map_vars_dagg
    m.map_vars_help = map_vars_help
    m.map_vars_smat = map_vars_smat

    logs[:mip_data] += toq()
    # println(model_mip)
end


#=========================================================
 Iterative algorithm functions
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    @printf "\nStarting iterative outer approximation algorithm:\n"
    soln_round_prev = fill(NaN, length(m.cols_bint))

    while true
        # Solve MIP model, finish if infeasible or unbounded, get objective
        tic()
        status_mip = solve(m.model_mip)
        logs[:mip_solve] += toq()
        if status_mip in (:Infeasible, :InfeasibleOrUnbounded)
            m.status = :Infeasible
            break
        end
        if status_mip == :Unbounded
            error("MIP solver returned status $status_mip, which could indicate that the cuts added were too weak\n")
        end
        m.obj_mip = getobjectivevalue(m.model_mip)

        # Reset cones summary values and calculate outer infeasibility of MIP solution
        reset_cone_summary!(m)
        calc_inf_outer!(m, logs)

        # Check for integer solutions repeating, finish if cycling
        bint_new = getvalue(m.x_bint)
        soln_round_curr = round(Int, bint_new)
        if soln_round_prev == soln_round_curr
            warn("Mixed-integer solutions are cycling; terminating Pajarito\n")
            m.status = :Suboptimal
            break
        end
        soln_round_prev = soln_round_curr

        # Solve conic subproblem given integer solution, add cuts to MIP, calculate cut and dual infeasibilities, save new solution if best objective
        process_conic!(m, bint_new, logs)

        # Calculate relative outer approximation gap, print gap and infeasibility statistics, finish if satisfy optimality gap condition
        m.gap_rel_opt = abs(m.obj_mip - m.obj_best) / (abs(m.obj_best) + 1e-5)
        print_gap(m, logs)
        print_inf(m)
        if m.gap_rel_opt < m.rel_gap
            m.status = :Optimal
            break
        end

        # Finish if exceeded timeout option
        if (time() - logs[:oa_alg]) > m.timeout
            m.status = :UserLimit
            break
        end
    end
end

# Solve the MIP model using MIP-solver-driven callback algorithm
function solve_mip_driven!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    # Initialize heuristic solution queue vectors, set bool to stop adding SOC cuts during MIP driven solve
    @printf "\nStarting MIP-solver-driven outer approximation algorithm:\n"
    m.bc_started = true
    m.queue_heur = Vector{Float64}[]

    # Add lazy cuts callback to solve the conic subproblem, add lazy cuts, and save a heuristic solution if conic solution is best
    function callback_lazy(cb)
        # Save callback reference so can use to adding lazy cuts
        m.model_mip.ext[:cb] = cb

        # Reset cones summary values and calculate outer infeasibility of MIP solution
        reset_cone_summary!(m)
        calc_inf_outer!(m, logs)

        # Solve conic subproblem given integer solution, add lazy cuts to MIP, calculate cut and dual infeasibilities, add solution to heuristic queue vectors if best objective
        process_conic!(m, getvalue(m.x_bint), logs)

        # Print cone infeasibilities
        print_inf(m)
    end
    addlazycallback(m.model_mip, callback_lazy)

    # Add heuristic callback to add each feasible solution from the current heuristic queue
    function callback_heur(cb)
        # Take each heuristic solution vector and add as a solution to the MIP
        tic()
        while !isempty(m.queue_heur)
            for (val, var) in zip(pop!(m.queue_heur), m.x_all)
                setsolutionvalue(cb, var, val)
            end

            addsolution(cb)
            logs[:n_sol_heur] += 1
        end
        logs[:cb_heur] += toq()
        logs[:n_cb_heur] += 1
    end
    addheuristiccallback(m.model_mip, callback_heur)

    # Start MIP solver
    logs[:mip_solve] = time()
    m.status = solve(m.model_mip)
    m.obj_mip = getobjectivevalue(m.model_mip)
    m.gap_rel_opt = abs(m.obj_mip - m.obj_best) / (abs(m.obj_best) + 1e-5)
    logs[:mip_solve] = time() - logs[:mip_solve]
end


#=========================================================
 Conic functions
=========================================================#

# Solve the initial conic relaxation model
function process_relax!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    # Initial conic relaxation data is
    # Min          c_orig*x_orig
    # st. b_orig - A_orig*x_orig in cone_con_orig
    #                     x_orig in cone_var_orig

    # Instantiate and solve the conic relaxation model
    tic()
    model_relax = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_relax, m.c_orig, m.A_orig, m.b_orig, m.cone_con_orig, m.cone_var_orig)
    MathProgBase.optimize!(model_relax)
    status_relax = MathProgBase.status(model_relax)
    logs[:relax_solve] += toq()

    # Only proceed if status is optimal or suboptimal, and print
    if status_relax == :Infeasible
        warn("Initial conic relaxation status was $status_relax\n")
        m.status = :Infeasible
        return false
    elseif status_relax == :Unbounded
        warn("Initial conic relaxation status was $status_relax\n")
        m.status = :InfeasibleOrUnbounded
        return false
    elseif !(status_relax in (:Optimal, :Suboptimal))
        error("Conic solver failure with status $status_relax\n")
    end
    if m.log_level > 0
        @printf "\nConic relaxation model solved:\n"
        @printf " - Status     = %14s\n" status_relax
        @printf " - Objective  = %14.6f\n" MathProgBase.getobjval(model_relax)
    end

    # Add initial dual cuts for each cone and calculate infeasibilities for cuts and duals
    tic()
    dual_con = MathProgBase.getdual(model_relax)
    dual_var = MathProgBase.getvardual(model_relax)

    # Add cuts for each nonlinear cone; if cone is a variable cone, use variable dual, else cone is a constraint cone so use constraint dual
    for n in 1:m.num_cone_nlnr
        add_cone_cuts!(m, n, m.map_spec[n], (m.map_ifvar[n] ? dual_var[m.map_ind[n]] : dual_con[m.map_ind[n]]))
    end
    logs[:relax_cuts] += toq()

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_relax)
        MathProgBase.freemodel!(model_relax)
    end

    return true
end

# Solve a conic subproblem given some solution to the integer variables
function process_conic!(m::PajaritoConicModel, bint_new::Vector{Float64}, logs::Dict{Symbol,Real})
    # Conic data is (constant soln_bint changes)
    # Min                               c_cont*x_cont
    # st. (b_full - A_bint*soln_bint) - A_cont*x_cont in cone_con_full
    #                                          x_cont in cone_var_cont

    # Calculate updated b vector for conic model from current integer solution
    tic()
    if any((val -> isnan(val)), bint_new)
        if m.mip_solver_drives
            println("Current integer solution vector has NaN values; terminating Pajarito\n")
            throw(CallbackAbort())
        else
            error("Current integer solution vector has NaN values; terminating Pajarito\n")
        end
    end
    b_conic = m.b_full - m.A_bint * bint_new

    # Instantiate and solve the conic model
    model_conic = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_conic, m.c_cont, m.A_cont, b_conic, m.cone_con_full, m.cone_var_cont)
    MathProgBase.optimize!(model_conic)
    status_conic = MathProgBase.status(model_conic)
    logs[:conic_solve] += toq()
    logs[:n_conic] += 1

    # Only proceed if status is infeasible, optimal or suboptimal
    if status_conic == :Unbounded
        if m.mip_solver_drives
            println("Conic status was $status_conic\n")
            throw(CallbackAbort())
        else
            error("Conic status was $status_conic\n")
        end
    elseif !(status_conic in (:Optimal, :Suboptimal, :Infeasible))
        if m.mip_solver_drives
            println("Conic solver failure with status $status_conic\n")
            throw(CallbackAbort())
        else
            error("Conic solver failure with status $status_conic\n")
        end
    end

    # Add dynamic cuts for each cone and calculate infeasibilities for cuts and duals
    tic()
    dual_con = MathProgBase.getdual(model_conic)
    dual_var = MathProgBase.getvardual(model_conic)

    # Add cuts for each nonlinear cone; if cone is a variable cone, use variable dual, else cone is a constraint cone so use constraint dual
    for n in 1:m.num_cone_nlnr
        # println("got dual $(m.map_spec[n]): $(m.map_ifvar[n] ? dual_var[m.map_ind_new[n]] : dual_con[m.map_ind_new[n]])")
        add_cone_cuts!(m, n, m.map_spec[n], (m.map_ifvar[n] ? dual_var[m.map_ind_new[n]] : dual_con[m.map_ind_new[n]]))
    end
    logs[:conic_cuts] += toq()

    # If feasible, check if new objective is best
    if status_conic != :Infeasible
        soln_new = MathProgBase.getsolution(model_conic)
        obj_new = dot(m.c_bint, bint_new) + dot(m.c_cont, soln_new)

        # If new objective is best, store new objective and solution
        if obj_new <= m.obj_best
            m.obj_best = obj_new
            m.soln_best[m.cols_bint] = bint_new
            m.soln_best[m.cols_cont] = soln_new

            # Construct MIP solutions on all variables in x_all, starting with integer then linear continuous variables then for each cone, the cone, disaggregated, and helper variables
            soln_all = vcat(bint_new, soln_new[m.oldnew_col[m.cols_linr]])
            for n in 1:m.num_cone_nlnr
                soln_cone = m.map_ifvar[n] ? soln_new[m.map_ind_new[n]] : (b_conic[m.map_ind_new[n]] - m.map_Asub[n] * soln_new)
                append!(soln_all, soln_cone)
                if !isempty(m.map_vars_dagg[n])
                    append!(soln_all, get_dagg_values(m.map_spec[n], soln_cone))
                end
                if !isempty(m.map_vars_help[n])
                    append!(soln_all, get_help_values(m.map_spec[n], soln_cone))
                end
            end
            @assert length(soln_all) == length(m.x_all)

            # Use soln_all to add a solution to the heuristic queue for MIP driven solve, or to warm-start the MIP for iterative algorithm
            if m.mip_solver_drives
                push!(m.queue_heur, soln_all)
            else
                for (val, var) in zip(soln_all, m.x_all)
                    setvalue(var, val)
                end
            end
        end
    end

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_conic)
        MathProgBase.freemodel!(model_conic)
    end
end

# Reset all summary values for all cones in preparation for next iteration
function reset_cone_summary!(m::PajaritoConicModel)
    if m.log_level > 1
        for spec_summ in values(m.summary)
            spec_summ[:outer_max_n] = 0
            spec_summ[:outer_max] = 0.
            spec_summ[:outer_min_n] = 0
            spec_summ[:outer_min] = 0.
            spec_summ[:dual_max_n] = 0
            spec_summ[:dual_max] = 0.
            spec_summ[:dual_min_n] = 0
            spec_summ[:dual_min] = 0.
            spec_summ[:cut_max_n] = 0
            spec_summ[:cut_max] = 0.
            spec_summ[:cut_min_n] = 0
            spec_summ[:cut_min] = 0.
        end
    end
end

# Calculate outer approximation infeasibilities for all nonlinear cones
function calc_inf_outer!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level > 1
        tic()
        for n in 1:m.num_cone_nlnr
            species = m.map_spec[n]
            soln = getvalue(m.map_vars[n])

            if species == :SOC
                inf_outer = sumabs2(soln[2:end]) - soln[1]^2

            elseif species == :SOCRotated
                inf_outer = sumabs2(soln[3:end]) - 2 * soln[1] * soln[2]

            elseif species == :ExpPrimal
                # TODO consider other case?
                inf_outer = soln[2] * exp(soln[1] / soln[2]) - soln[3]

            elseif species == :SDP
                inf_outer = -eigmin(make_smat(soln))
            end

            if inf_outer > 0.
                m.summary[species][:outer_max_n] += 1
                m.summary[species][:outer_max] = max(inf_outer, m.summary[species][:outer_max])
            elseif inf_outer < 0.
                m.summary[species][:outer_min_n] += 1
                m.summary[species][:outer_min] = max(-inf_outer, m.summary[species][:outer_min])
            end
        end
        logs[:outer_inf] += toq()
    end
end


#=========================================================
 Cut adding functions
=========================================================#

# Process dual vector and add dual cuts to MIP
function add_cone_cuts!(m::PajaritoConicModel, n::Int, species::Symbol, dual::Vector{Float64})
    # Rescale the dual, don't add zero vectors
    if maximum(abs(dual)) > m.zero_tol
        dual = dual ./ maximum(abs(dual))
    else
        return
    end

    # Sanitize rescaled dual: remove near-zeros
    dual[abs(dual) .< m.zero_tol] = 0.

    # For primal cone species:
    # 1 - calculate dual cone infeasibility of dual vector (negative value means strictly feasible in cone, zero means on cone boundary, positive means infeasible for cone)
    # 2 - process dual (project if necessary, depending on dual inf), dropping any trivial/zero cuts
    # 3 - disaggregate cut(s) if necessary
    # 4 - add dual cut(s)
    if species == :SOC
        inf_dual = sumabs2(dual[2:end]) - dual[1]^2

        if (dual[1] > 0.) && (!m.drop_dual_infeas || (inf_dual <= 0.))
            if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
                # Epigraph variable equals norm
                dual = vcat(norm(dual[2:end]), dual[2:end])
            end

            if isempty(m.map_vars_dagg[n])
                # Nondisaggregated cuts
                add_linear_cut!(m, m.summary[species], m.map_vars[n], dual)
            else
                # Disaggregated cuts
                for ind in eachindex(m.map_vars_dagg[n])
                    add_linear_cut!(m, m.summary[species], [m.map_vars[n][1], m.map_vars_dagg[n][ind], m.map_vars[n][(1 + ind)]], [(dual[(1 + ind)] / dual[1])^2, 1., (2 * dual[(1 + ind)] / dual[1])])
                end
            end
        end

    elseif species == :SOCRotated
        inf_dual = sumabs2(dual[3:end]) - 2 * dual[1] * dual[2]

        if (dual[1] > 0.) && (dual[2] > 0.) && (!m.drop_dual_infeas || (inf_dual <= 0.))
            if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
                # Rescale variables in bilinear term
                dual = vcat((dual[1:2] * (norm(dual[3:end]) / sqrt(2 * dual[1] * dual[2]))), dual[3:end])
            end

            # TODO when disagg_soc enabled for SOCRotated, add two sets of cuts
            # if isempty(m.map_vars_dagg[n])
            add_linear_cut!(m, m.summary[species], m.map_vars[n], dual)
            # else
            #     for ind in eachindex(m.map_vars_dagg[n])
            #         add_linear_cut!(m, m.summary[species], [m.map_vars[n][1], m.map_vars_dagg[n][ind], m.map_vars[n][(2 + ind)]], [dual[1], (dual[(2 + ind)]^2 / dual[2]), dual[(2 + ind)]])
            #     end
            # end
        end

    elseif species == :ExpPrimal
        # TODO use which definition of dual cone - log space or exp space (change MathProgBase conic definition)
        # Do not add cuts for dual[1] >= 0 because these simply enforce the nonnegativity of x[2] and x[3] in ExpPrimal
        # Do not add cuts for dual[3] < 0 because can't project onto dual[3] = 0
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
            # Exp space definition
            inf_dual = -dual[1] * exp(dual[2] / dual[1]) - e * dual[3]

            # Log space definition
            # inf_dual = -dual[1] * log(-dual[1] / dual[3]) + dual[1] - dual[2]

            if !m.drop_dual_infeas || (inf_dual <= 0.)
                if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
                    # Epigraph variable equals LHS
                    dual = vcat(dual[1], dual[2], (-dual[1] * exp(dual[2] / dual[1])) / e)
                end

                add_linear_cut!(m, m.summary[species], m.map_vars[n], dual)
            end
        end

    elseif species == :SDP
        # Get eigendecomposition
        (eigvals_dual, eigvecs_dual) = eig(make_smat(dual))

        inf_dual = -minimum(eigvals_dual)

        # If using eigenvector cuts, add SOC or linear eig cuts, else add linear cut
        if m.sdp_eig
            # Get array of (orthonormal) eigenvector columns with significant nonnegative eigenvalues, and sanitize eigenvectors
            Vdual = eigvecs_dual[:, (eigvals_dual .>= m.sdp_tol_eigval)]
            Vdual[abs(Vdual) .< m.sdp_tol_eigvec] = 0.

            if size(Vdual, 2) > 0
                # Cannot add SOC cuts during MIP solve
                if m._sdp_soc && !m.bc_started
                    # add_sdp_soc_cuts!(m, n, m.summary[species], m.map_vars_smat[n], Vdual)
                    add_sdp_soc_cuts!(m, n, m.summary[species], m.map_vars_smat[n], Vdual)
                else
                    # Add linear cut for each significant eigenvector
                    for jV in 1:size(Vdual, 2)
                        # println(vec(Vdual[:, jV] * Vdual[:, jV]'))
                        # add_linear_cut!(m, m.summary[species], vec(m.map_vars_smat[n]), vec(Vdual[:, jV] * Vdual[:, jV]'))
                        add_linear_cut!(m, m.summary[species], m.map_vars[n], make_svec(Vdual[:, jV] * Vdual[:, jV]'))
                    end
                end
            end

        elseif any(eigvals_dual .>= 0.) && (!m.drop_dual_infeas || (inf_dual <= 0.))
            if (inf_dual > 0.) && m.proj_dual_infeas
                # Project by taking sum of nonnegative eigenvalues times outer products of corresponding eigenvectors
                # dual_smat = sum([(eigvals_dual[jV] * (eigvecs_dual[:, jV] * eigvecs_dual[:, jV]')) for jV in find((val -> val >= 0.), eigvals_dual)])
                # dual = make_svec(sum([(eigvals_dual[jV] * (eigvecs_dual[:, jV] * eigvecs_dual[:, jV]')) for jV in find((val -> val > 0.), eigvals_dual)]))

                # Re-sanitize and add cut in smat space
                # dual_smat[abs(dual_smat) .< m.zero_tol] = 0.
                # add_linear_cut!(m, m.summary[species], vec(m.map_vars_smat[n]), vec(dual_smat))
                # add_linear_cut!(m, m.summary[species], m.map_vars[n], make_svec(Vdual[:, jV] * Vdual[:, jV]'))

                eigvals_dual[eigvals_dual .<= 0.] = 0.
                dual = make_svec(eigvecs_dual * diagm(eigvals_dual) * eigvecs_dual')
            end

            add_linear_cut!(m, m.summary[species], m.map_vars[n], dual)
        end
    end

    # Update dual infeasibility
    if m.log_level > 1
        if inf_dual > 0.
            m.summary[species][:dual_max_n] += 1
            m.summary[species][:dual_max] = max(inf_dual, m.summary[species][:dual_max])
        elseif inf_dual < 0.
            m.summary[species][:dual_min_n] += 1
            m.summary[species][:dual_min] = max(-inf_dual, m.summary[species][:dual_min])
        end
    end
end

# Add a single linear cut and calculate cut infeasibility
function add_linear_cut!(m::PajaritoConicModel, spec_summ::Dict{Symbol,Real}, vars::Vector{JuMP.Variable}, cut::Vector{Float64})
    # Add cut (lazy cut if using MIP driven solve)
    if haskey(m.model_mip.ext, :cb)
        @lazyconstraint(m.model_mip.ext[:cb], dot(cut, vars) >= 0.)
    else
        @constraint(m.model_mip, dot(cut, vars) >= 0.)
    end

    # Update cut infeasibility
    if m.log_level > 1
        inf_cut = -dot(cut, getvalue(vars))
        if inf_cut > 0.
            spec_summ[:cut_max_n] += 1
            spec_summ[:cut_max] = max(inf_cut, spec_summ[:cut_max])
        elseif inf_cut < 0.
            spec_summ[:cut_min_n] += 1
            spec_summ[:cut_min] = max(-inf_cut, spec_summ[:cut_min])
        end
    end
end

# Add SDP SOC cuts (derived from Schur complement) for each eigenvector and each diagonal element
function add_sdp_soc_cuts!(m::PajaritoConicModel, n::Int, spec_summ::Dict{Symbol,Real}, vars_smat::Array{JuMP.Variable,2}, Vdual::Array{Float64,2})
    (nSD, nV) = size(Vdual)

    # For each sanitized eigenvector with significant eigenvalue
    for jV in 1:nV
        # Get eigenvector and form rank-1 outer product
        vj = Vdual[:, jV]
        vvj = vj * vj'

        # For each diagonal element of SDP
        for iSD in 1:nSD
            no_i = vcat(1:(iSD - 1), (iSD + 1):nSD)

            # Add helper variable for subvector iSD product
            @variable(m.model_mip, vx, basename="h$(n)SDvx_$(jV)_$(iSD)")
            @constraint(m.model_mip, vx == vecdot(vj[no_i], vars_smat[no_i, iSD]))

            # Add helper variable for submatrix iSD product
            @variable(m.model_mip, vvX >= 0., basename="h$(n)SDvvX_$(jV)_$(iSD)")
            @constraint(m.model_mip, vvX == vecdot(vvj[no_i, no_i], vars_smat[no_i, no_i]))

            # Add SOC constraint
            @constraint(m.model_mip, vars_smat[iSD, iSD] * vvX >= vx^2)

            # Update cut infeasibility
            if m.log_level > 1
                inf_cut = -getvalue(vars_smat[iSD, iSD]) * vecdot(vvj[no_i, no_i], getvalue(vars_smat[no_i, no_i])) + (vecdot(vj[no_i], getvalue(vars_smat[no_i, iSD])))^2
                if inf_cut > 0.
                    spec_summ[:cut_max_n] += 1
                    spec_summ[:cut_max] = max(inf_cut, spec_summ[:cut_max])
                elseif inf_cut < 0.
                    spec_summ[:cut_min_n] += 1
                    spec_summ[:cut_min] = max(-inf_cut, spec_summ[:cut_min])
                end
            end
        end
    end
end


#=========================================================
 Algorithm utilities
=========================================================#

# Use vector on nondisaggregated variables to calculate vector for disaggregated MIP variables
function get_dagg_values(species::Symbol, vals::Vector{Float64})
    if species == :SOC
        if vals[1] == 0.
            return zeros(length(vals) - 1)
        else
            return (vals[2:end]).^2 ./ vals[1]
        end

    # TODO this is only for dividing by second element - update for both
    # elseif species == :SOCRotated
    #     if (vals[1] == 0.) || (vals[2] == 0.)
    #         return zeros(length(vals) - 2)
    #     else
    #         return (vals[3:end]).^2 ./ vals[2]
    #     end
    end
end

# Use vector on nondisaggregated variables to calculate vector for helper MIP variables
function get_help_values(species::Symbol, vals::Vector{Float64})
    if species == :SOCRotated
        return [2 * vals[2]]

    elseif species == :SDP
        nSD = round(Int, sqrt(1/4 + 2 * length(vals)) - 1/2)
        help_cone = Vector{Float64}(nSD)
        kSD = 1
        for jSD in 1:nSD, iSD in jSD:nSD
            if jSD == iSD
                help_cone[jSD] = vals[kSD] * sqrt(2)
            end
            kSD += 1
        end

        return help_cone
    end
end

# Transform an svec form into an smat form
function make_smat(svec::Vector{Float64})
    nSD = round(Int, sqrt(1/4 + 2 * length(svec)) - 1/2)
    smat = Array{Float64,2}(nSD, nSD)
    kSD = 1
    for jSD in 1:nSD, iSD in jSD:nSD
        if jSD == iSD
            smat[iSD, jSD] = svec[kSD]
        else
            smat[iSD, jSD] = smat[jSD, iSD] = svec[kSD] / sqrt(2)
        end
        kSD += 1
    end

    return smat
end

# Transform an smat form into an svec form
function make_svec(smat::Array{Float64,2})
    nSD = size(smat, 2)
    svec = Vector{Float64}(round(Int, (nSD * (nSD + 1) / 2)))
    kSD = 1
    for jSD in 1:nSD, iSD in jSD:nSD
        if jSD == iSD
            svec[kSD] = smat[iSD, jSD]
        else
            svec[kSD] = smat[iSD, jSD] * sqrt(2)
        end
        kSD += 1
    end

    return svec
end


#=========================================================
 Logging, printing, testing functions
=========================================================#

# Create dictionary of logs for timing and iteration counts
function create_logs()
    logs = Dict{Symbol,Real}()

    # Timers
    logs[:conic_data] = 0.  # Generating conic data
    logs[:mip_data] = 0.    # Generating MIP data
    logs[:total_setup] = 0. # Performing total optimize algorithm
    logs[:relax_solve] = 0. # Solving initial conic relaxation model
    logs[:relax_cuts] = 0.  # Adding cuts for initial relaxation model
    logs[:conic_solve] = 0. # Solving conic subproblem model
    logs[:conic_cuts] = 0.  # Adding cuts for conic subproblem model
    logs[:outer_inf] = 0.   # Calculating outer infeasibility for all cones
    logs[:cb_heur] = 0.     # Using heuristic callback (MIP driven solve only)
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:oa_alg] = 0.      # Performing outer approximation algorithm

    # Iteration counters
    logs[:n_conic] = 0      # Number of conic subproblem solves
    logs[:n_sol_heur] = 0   # Total number of heuristic solutions added to MIP
    logs[:n_cb_heur] = 0    # Number of times heuristic callback is called

    return logs
end

# Print cone dimensions summary
function print_cones(m::PajaritoConicModel, summary::Dict{Symbol,Dict{Symbol,Real}})
    if m.log_level > 0
        @printf "\n%-10s | %-8s | %-8s | %-8s\n" "Species" "Count" "Min dim" "Max dim"
        for (species, spec_summ) in summary
            @printf "%10s | %8d | %8d | %8d\n" species spec_summ[:count] spec_summ[:min_dim] spec_summ[:max_dim]
        end

        @printf "\n"
        flush(STDOUT)
    end
end

# Print dual infeasibility only
function print_inf_dual(m::PajaritoConicModel)
    if m.log_level > 1
        @printf "\n%-10s | %-32s\n" "Species" "Dual cone infeas"
        @printf "%-10s | %-6s %-8s  %-6s %-8s\n" "" "Inf" "Worst" "Feas" "Worst"
        for (species, spec_summ) in m.summary
            @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" species spec_summ[:dual_max_n] spec_summ[:dual_max] spec_summ[:dual_min_n] spec_summ[:dual_min]
        end

        @printf "\n"
        flush(STDOUT)
    end
end

# Print cones infeasibilities
function print_inf(m::PajaritoConicModel)
    if m.log_level > 1
        @printf "\n%-10s | %-32s | %-32s | %-32s\n" "Species" "Outer approx infeas" "Dual cone infeas" "Cut infeas"
        @printf "%-10s | %-6s %-8s  %-6s %-8s | %-6s %-8s  %-6s %-8s | %-6s %-8s  %-6s %-8s\n" "" "Inf" "Worst" "Feas" "Worst" "Inf" "Worst" "Feas" "Worst" "Inf" "Worst" "Feas" "Worst"
        for (species, spec_summ) in m.summary
            @printf "%10s | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e\n" species spec_summ[:outer_max_n] spec_summ[:outer_max] spec_summ[:outer_min_n] spec_summ[:outer_min] spec_summ[:dual_max_n] spec_summ[:dual_max] spec_summ[:dual_min_n] spec_summ[:dual_min] spec_summ[:cut_max_n] spec_summ[:cut_max] spec_summ[:cut_min_n] spec_summ[:cut_min]
        end

        @printf "\n"
        flush(STDOUT)
    end
end

# Print objective gap information
function print_gap(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level > 0
        if (logs[:n_conic] == 0) || (m.log_level > 1)
            @printf "\n%-4s | %-14s | %-14s | %-11s | %-11s\n" "Iter" "Best obj" "OA obj" "Rel gap" "Time (s)"
        end

        if m.gap_rel_opt < 1000
            @printf "%4d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" logs[:n_conic] m.obj_best m.obj_mip m.gap_rel_opt (time() - logs[:oa_alg])
        else
            @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" logs[:n_conic] m.obj_best m.obj_mip ">1000" (time() - logs[:oa_alg])
        end

        flush(STDOUT)
    end
end

# Save solution data to file
# function save_finish(m::PajaritoConicModel, logs::Dict{Symbol,Real})
#     if !isempty(m.path)
#         @printf "\nWriting results to file %s... " m.path
#
#         out_file = open("output.txt", "a")
#
#         write(out_file, "$(m.path):\n$(m.status)\n$(logs[:n_conic])\n$(logs[:total_setup]) $(logs[:oa_alg]) $(logs[:mip_solve]) $(logs[:conic_solve])\n$(m.obj_best) $(m.obj_mip) $(m.gap_rel_opt)\n$(m.soln_best)\n")
#
#         close(out_file)
#
#         @printf "save complete.\n\n"
#         flush(STDOUT)
#     end
# end

# Print after finish
function print_finish(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.mip_solver_drives
        @printf "\nFinished MIP-solver-driven outer approximation algorithm:\n"
    else
        @printf "\nFinished iterative outer approximation algorithm:\n"
    end
    @printf " - Total time (s)       = %14.2e\n" (logs[:total_setup] + logs[:oa_alg])
    @printf " - Status               = %14s\n" m.status
    @printf " - Best feasible obj.   = %+14.6e\n" m.obj_best
    @printf " - Final OA obj. bound  = %+14.6e\n" m.obj_mip
    @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt
    @printf " - Conic iter. count    = %14d\n" logs[:n_conic]

    if m.log_level > 1
        if m.mip_solver_drives
            @printf " - Heur. callback count = %14d\n" logs[:n_cb_heur]
            @printf " - Heur. solution count = %14d\n" logs[:n_sol_heur]
        end

        @printf "\nTimers (s):\n"
        @printf " - Setup                = %14.2e\n" logs[:total_setup]
        if m.log_level > 1
            @printf " -- Create conic data   = %14.2e\n" logs[:conic_data]
            @printf " -- Create MIP data     = %14.2e\n" logs[:mip_data]
            @printf " -- Solve relax         = %14.2e\n" logs[:relax_solve]
            @printf " -- Add relax cuts      = %14.2e\n" logs[:relax_cuts]
        end
        if m.mip_solver_drives
            @printf " - MIP-driven algorithm = %14.2e\n" logs[:oa_alg]
        else
            @printf " - Iterative algorithm  = %14.2e\n" logs[:oa_alg]
        end
        @printf " -- Solve MIP           = %14.2e\n" logs[:mip_solve]
        @printf " -- Solve conic         = %14.2e\n" logs[:conic_solve]
        if m.log_level > 1
            @printf " -- Add conic cuts      = %14.2e\n" logs[:conic_cuts]
            @printf " -- Calc. outer inf.    = %14.2e\n" logs[:outer_inf]
            if m.mip_solver_drives
                @printf " -- Use heur. callback  = %14.2e\n" logs[:cb_heur]
            end
        end
    end

    @printf "\n"
    flush(STDOUT)
end
