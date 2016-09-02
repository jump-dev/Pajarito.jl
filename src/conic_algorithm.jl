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
- when JuMP can handle anonymous variables without errors, use that syntax
- use new JuMP MPB time limit parameter for solver
- does partial warm-start work? maybe need to extend to full MIP solution
- add initial LINEAR sdp cuts (redundant with initial SOC cuts) -2m_ij <= m_ii + m_jj, 2m_ij <= m_ii + m_jj, all i,j
- update for 0.5
- replace "for i in 1:..."
- want to be able to query logs information etc
- use option for JP updated SOC disagg_soc with half as many cuts, use abs value variable
- have option to only add violated cuts (especially for SDP, where each SOC cut slows down mip and we have many SOC cuts)
- print cone info to one file and gap info to another file
- what to do if experience conic problem strong duality failure? could use a no-good cut on that integer solution and proceed, but that could cut off optimal sol?
- dual cone projection - implement multiple heuristic projections and optimal euclidean projection
- currently all SDP sanitized eigvecs have norm 1, but may want to multiply V by say 100 (or perhaps largest eigenvalue) before removing zeros, to get more significant digits
=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
    log_level::Int              # Verbosity flag: 1 for minimal OA iteration and solve statistics, 2 for including cone summary information, 3 for running commentary
    mip_solver_drives::Bool     # Let MIP solver manage convergence and conic subproblem calls (to add lazy cuts and heuristic solutions in branch and cut fashion)
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    disagg_soc::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
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
    var_types::Vector{Symbol}   # Variable types vector on original variables (only :Bin, :Cont, :Int)
    var_start::Vector{Float64}  # Variable warm start vector on original variables

    # Transformed data
    cone_con_new::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data after converting variable cones to constraint cones
    cone_var_new::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data after converting variable cones to constraint cones and adding slacks
    num_con_new::Int            # Number of constraints after converting variable cones to constraint cones
    num_var_new::Int            # Number of variables after adding slacks
    b_new::Vector{Float64}      # Subvector of b containing full rows
    c_new::Vector{Float64}      # Objective coefficient subvector for continuous variables
    A_new::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    row_to_slckj::Dict{Int,Int} # Dictionary from row index in MIP to slack column index if slack exists
    row_to_slckv::Dict{Int,Float64} # Dictionary from row index in MIP to slack coefficient if slack exists
    num_cone_nlnr::Int          # Number of nonlinear cones

    # MIP constructed data
    model_mip::JuMP.Model       # JuMP MIP (outer approximation) model
    x_mip::Vector{JuMP.Variable} # JuMP vector of original variables
    map_spec::Vector{Symbol}    # Species of nonlinear cone
    map_rows::Vector{Vector{Int}} # Row indices in MIP for nonlinear cone
    map_dim::Vector{Int}        # Dimension of nonlinear cone
    map_vars::Vector{Vector{Vector{JuMP.Variable}}} # JuMP variables associated with slacks of nonlinear cone
    map_coefs::Vector{Vector{Float64}} # Coefficients associated with slacks of nonlinear cone (possibly not 1 for original slacks)
    map_isnew::Vector{Vector{Bool}} # Vector of bools indicating which variables in map_vars were added to the MIP

    # Conic constructed data
    cone_con_sub::Vector{Tuple{Symbol,Vector{Int}}} # Constraint cones data in conic subproblem
    cone_var_sub::Vector{Tuple{Symbol,Vector{Int}}} # Variable cones data in conic subproblem
    map_rows_sub::Vector{Vector{Int}} # Row indices in conic subproblem for nonlinear cone
    cols_cont::Vector{Int}      # Column indices of continuous variables in MIP
    cols_int::Vector{Int}       # Column indices of integer variables in MIP
    A_sub_cont::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and continuous variable columns
    A_sub_int::SparseMatrixCSC{Float64,Int64} # Submatrix of A containing full rows and integer variable columns
    b_sub::Vector{Float64}      # Subvector of b containing full rows
    c_sub_cont::Vector{Float64} # Subvector of c for continuous variables
    c_sub_int::Vector{Float64}  # Subvector of c for integer variables

    # Dynamic solve data
    summary::Dict{Symbol,Dict{Symbol,Real}} # Infeasibilities (outer, cut, dual) of each cone species at current iteration
    bc_started::Bool            # Bool for whether MIP-driven solve has begun
    status::Symbol              # Current solve status
    obj_mip::Float64            # Latest MIP (outer approx) objective value
    obj_best::Float64           # Best conic (feasible) objective value
    gap_rel_opt::Float64        # Relative optimality gap = |obj_mip - obj_best|/|obj_best|
    soln_best::Vector{Float64}  # Best original solution vector (corresponding to best objective)
    queue_heur::Vector{Vector{Vector{Float64}}} # Heuristic queue for MIP solutions (if mip_solver_drives)

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
            mip_spec = MathProgBase.supportedcones(m.mip_solver)
            if ((:SOC in mip_spec) && (:SOCRotated in mip_spec))
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
        m.var_types = Symbol[]
        m.var_start = Float64[]
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
        if spec == :Zero
            error("A cone $spec is in the variable cones\n")
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
        conic_spec = MathProgBase.supportedcones(m.cont_solver)
        for (spec, inds) in vcat(cone_con, cone_var)
            if !(spec in conic_spec)
                error("Cones $spec are not supported by the specified conic solver\n")
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
        @printf "Matrix A has %d entries smaller than zero tolerance %e; performance may be improved by first fixing small magnitudes to zero\n" A_num_zeros m.zero_tol
    end

    m.num_con_orig = num_con_orig
    m.num_var_orig = num_var_orig
    m.A_orig = A_sp
    m.c_orig = c
    m.b_orig = b
    m.cone_con_orig = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_con]
    m.cone_var_orig = Tuple{Symbol,Vector{Int}}[(spec, collect(inds)) for (spec, inds) in cone_var]
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
function MathProgBase.setwarmstart!(m::PajaritoConicModel, var_start::Vector{Real})
    # Check if vector can be loaded
    if m.status != :Loaded
        error("Must specify warm start right after loading problem\n")
    end
    if length(var_start) != m.num_var_orig
        error("Warm start vector length ($(length(var_start))) does not match number of variables ($(m.num_var_orig))\n")
    end

    m.var_start = var_start
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
    logs[:total_setup] = time()
    trans_data!(m, logs)
    print_cones(m, m.summary)
    create_mip_data!(m, logs)
    create_conic_data!(m, logs)
    reset_cone_summary!(m)

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
MathProgBase.getobjbound(m::PajaritoConicModel) = m.obj_mip
MathProgBase.getsolution(m::PajaritoConicModel) = m.soln_best


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
    num_var_new = m.num_var_orig
    (A_I, A_J, A_V) = findnz(m.A_orig)

    for (spec, cols) in m.cone_var_orig
        if spec == :Zero
            error("Decide what to do with zero cone variables")
        elseif spec in (:Free, :NonNeg, :NonPos)
            push!(cone_var_new, (spec, cols))
        else
            push!(cone_var_new, (:Free, cols))
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

    # Convert SOCRotated cones to SOC cones
    # (y,z,x) in RSOC <=> (y+z,y-z,sqrt(2)*x) in SOC
    socr_rows = Vector{Int}[]
    for n_cone in 1:length(cone_con_new)
        (spec, rows) = cone_con_new[n_cone]
        if spec == :SOCRotated
            cone_con_new[n_cone] = (:SOC, rows)
            push!(socr_rows, rows)
        end
    end

    for rows in socr_rows
        inds_1 = (A_I .== rows[1])
        inds_2 = (A_I .== rows[2])
        inds_3 = find(i -> (i in rows[3:end]), A_I)

        append!(A_I, fill(rows[1], length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])

        append!(A_I, fill(rows[2], length(inds_1)))
        append!(A_J, A_J[inds_1])
        append!(A_V, -A_V[inds_1])

        for ind in inds_3
            A_V[ind] = sqrt(2) * A_V[ind]
        end

        # row_y = A_new[rows[1], :]
        # row_z = A_new[rows[2], :]
        # rows_x = A_new[rows[3:end], :]
        #
        # A_new[rows[1], :] = row_y .+ row_z
        # A_new[rows[2], :] = row_y .- row_z
        # A_new[rows[3:end], :] = sqrt(2) .* rows_x
        #
        # cone_con_new[n_cone] = (:SOC, rows)
    end

    # Detect existing slack variables in nonlinear cone rows with b=0, corresponding to isolated row nonzeros equal to -1
    row_slck_count = zeros(Int, num_con_new)
    for (ind, i) in enumerate(A_I)
        if (b_new[i] == 0.) && (A_V[ind] != 0.)
            if row_slck_count[i] == 0
                row_slck_count[i] = ind
            elseif row_slck_count[i] > 0
                row_slck_count[i] == -1
            end
        end
    end

    row_to_slckj = Dict{Int,Int}()
    row_to_slckv = Dict{Int,Float64}()

    slack_tol = 0.1 # TODO maybe make this an option; may also want upper bound

    summary = Dict{Symbol,Dict{Symbol,Real}}()

    num_cone_nlnr = 0
    for (spec, rows) in cone_con_new
        if !(spec in (:Free, :Zero, :NonNeg, :NonPos))
            num_cone_nlnr += 1

            # Create cone summary dictionary
            if !haskey(summary, spec)
                summary[spec] = Dict{Symbol,Real}(:count => 1, :max_dim => length(rows), :min_dim => length(rows))
            else
                summary[spec][:count] += 1
                if (summary[spec][:max_dim] < length(rows))
                    summary[spec][:max_dim] = length(rows)
                elseif (summary[spec][:min_dim] > length(rows))
                    summary[spec][:min_dim] = length(rows)
                end
            end

            # Use a tolerance for abs of the coefficient; if too small, cannot use as a slack variable
            for i in rows
                if row_slck_count[i] > 0
                    if abs(A_V[row_slck_count[i]]) > slack_tol
                    # if A_V[row_slck_count[i]] == -1.
                        row_to_slckj[i] = A_J[row_slck_count[i]]
                        row_to_slckv[i] = A_V[row_slck_count[i]]
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
    m.b_new = b_new
    m.c_new = m.c_orig
    m.A_new = sparse(A_I, A_J, A_V, num_con_new, num_var_new)
    m.row_to_slckj = row_to_slckj
    m.row_to_slckv = row_to_slckv
    m.num_cone_nlnr = num_cone_nlnr
    m.summary = summary

    logs[:trans_data] += toq()
end

# Generate MIP model and maps relating conic model and MIP model variables
function create_mip_data!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Initialize JuMP model for MIP outer approximation problem
    model_mip = JuMP.Model(solver = m.mip_solver)

    x_mip = @variable(model_mip, [1:m.num_var_new])

    for j in 1:m.num_var_new
        setcategory(x_mip[j], m.var_types[j])
    end

    # TODO warm start entire algorithm
    # if !isempty(m.var_start)
    #     for j in 1:m.num_var_new
    #         setvalue(x_mip[j], m.var_start[j])
    #     end
    # end

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
            error("Decide what to do with zero cone variables") #TODO remove
        end
    end

    # Create maps from nonlinear cone index to associated information and slacks
    map_spec = Vector{Symbol}(m.num_cone_nlnr)
    map_rows = Vector{Vector{Int}}(m.num_cone_nlnr)
    map_dim = Vector{Int}(m.num_cone_nlnr)
    map_vars = Vector{Vector{Vector{JuMP.Variable}}}(m.num_cone_nlnr)
    map_coefs = Vector{Vector{Float64}}(m.num_cone_nlnr)
    map_isnew = Vector{Vector{Bool}}(m.num_cone_nlnr)

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    lhs_expr = m.b_new - m.A_new * x_mip

    n_cone = 0
    for (spec, rows) in m.cone_con_new
        if spec == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0.)

        elseif spec == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)

        elseif spec == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)

        else
            # Create slack vector for nonlinear cone, re-using any slacks detected earlier
            n_cone += 1
            map_spec[n_cone] = spec
            map_rows[n_cone] = rows

            len = length(rows)
            vars = Vector{JuMP.Variable}(len)
            coefs = ones(len)
            isnew = Vector{Bool}(len)
            for (ind, i) in enumerate(rows)
                if haskey(m.row_to_slckj, i)
                    vars[ind] = x_mip[m.row_to_slckj[i]]
                    setname(x_mip[m.row_to_slckj[i]], "v$(m.row_to_slckj[i])_s$(i)_c$(n_cone)")
                    coefs[ind] = - m.row_to_slckv[i]
                    isnew[ind] = false
                else
                    vars[ind] = @variable(model_mip, _, basename="s$(i)_c$(n_cone)", start=0.)
                    @constraint(model_mip, lhs_expr[i] - vars[ind] == 0.)
                    isnew[ind] = true
                end
            end
            map_coefs[n_cone] = coefs
            map_isnew[n_cone] = isnew
            map_vars[n_cone] = Vector{JuMP.Variable}[vars]

            # Set bounds on variables and save dimensions, add additional constraints/variables
            if spec == :SOC
                map_dim[n_cone] = len

                if sign(coefs[1]) == 1
                    setlowerbound(vars[1], 0.)
                else
                    setupperbound(vars[1], 0.)
                end

                if m._soc_in_mip
                    #TODO use norm, fix jump issue 784 so that warm start works
                    error("SOC in MIP is currently broken; terminating Pajarito")
                    # @constraint(model_mip, norm2{coefs[j] .* vars[j], j in 2:len} <= coefs[1] * vars[1])

                elseif m.disagg_soc && (len > 2)
                    dagg = @variable(model_mip, [j in 1:(len - 1)], lowerbound=0., basename="d$(n_cone)SOC", start=0.)
                    @constraint(model_mip, coefs[1] * vars[1] - sum(dagg) >= 0.)
                    push!(map_vars[n_cone], dagg)
                end

            elseif spec == :ExpPrimal
                map_dim[n_cone] = len

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

            elseif spec == :SDP
                # Set up svec space variable vector
                nSD = round(Int, sqrt(1/4 + 2 * len) - 1/2)
                map_dim[n_cone] = nSD

                kSD = 1
                for jSD in 1:nSD, iSD in jSD:nSD
                    if jSD == iSD
                        if sign(coefs[kSD]) == 1
                            setlowerbound(vars[kSD], 0.)
                        else
                            setupperbound(vars[kSD], 0.)
                        end
                    end
                    kSD += 1
                end

                # Add initial SDP linear cuts
                # TODO

                # Set up helper variables and initial SDP SOC cuts
                # TODO rethink helper and smat variables
                # TODO not using coefs properly
                if m._sdp_init_soc || m._sdp_soc
                    error("SOC in MIP is currently broken; terminating Pajarito")

                    # # Set up smat space variable array and add optional SDP initial SOC and dynamic SOC cuts
                    # help = @variable(model_mip, [j in 1:nSD], lowerbound=0., basename="h$(n_cone)SDhelp", start=0.)
                    # smat = Array{JuMP.AffExpr,2}(nSD, nSD)
                    # kSD = 1
                    # for jSD in 1:nSD, iSD in jSD:nSD
                    #     if jSD == iSD
                    #         # Create helper variable corresponding to diagonal element multiplied by sqrt(2)
                    #         @constraint(model_mip, help[jSD] == coefs[kSD] * sqrt(2) * vars[kSD])
                    #         smat[jSD, iSD] = help[jSD]
                    #     else
                    #         smat[jSD, iSD] = smat[iSD, jSD] = coefs[kSD] * vars[kSD]
                    #         if m._sdp_init_soc
                    #             # Add initial rotated SOC for off-diagonal element to enforce 2x2 principal submatrix PSDness
                    #             # TODO this won't work: not intepreted as rsoc, may need manual SOC transformation, norm maybe?
                    #             @constraint(model_mip, help[jSD] * help[iSD] >= (coefs[kSD] * vars[kSD])^2)
                    #         end
                    #     end
                    #     kSD += 1
                    # end
                    # push!(map_vars[n_cone], help)
                    # push!(map_vars[n_cone], smat)
                end
            end
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_mip = x_mip
    m.map_spec = map_spec
    m.map_rows = map_rows
    m.map_dim = map_dim
    m.map_vars = map_vars
    m.map_coefs = map_coefs
    m.map_isnew = map_isnew

    logs[:mip_data] += toq()
    # println(model_mip)
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
            if m.var_types[j] == :Cont
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
    map_rows_sub = Vector{Int}[]

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
            push!(map_rows_sub, collect((num_full + 1):(num_full + length(rows))))
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

    logs[:conic_data] += toq()
end


#=========================================================
 Iterative algorithm functions
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level > 0
        @printf "\nStarting iterative outer approximation algorithm:\n"
    end
    soln_int_prev = fill(NaN, length(m.cols_int))

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
        soln_int = getvalue(m.x_mip[m.cols_int])
        soln_int_curr = round(Int, soln_int)
        if soln_int_prev == soln_int_curr
            # Check if we've converged anyway
            # TODO gap logic? and avoid doing this twice
            m.gap_rel_opt = abs(m.obj_mip - m.obj_best) / (abs(m.obj_best) + 1e-5)
            if m.gap_rel_opt < m.rel_gap
                m.status = :Optimal
            else
                warn("Mixed-integer solutions are cycling; terminating Pajarito\n")
                m.status = :Suboptimal
            end
            break
        end
        soln_int_prev = soln_int_curr

        # Solve conic subproblem given integer solution, add cuts to MIP, calculate cut and dual infeasibilities, save new solution if best objective
        process_conic!(m, soln_int, logs)

        # Calculate relative outer approximation gap, print gap and infeasibility statistics, finish if satisfy optimality gap condition
        # TODO gap logic?
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
    if m.log_level > 0
        @printf "\nStarting MIP-solver-driven outer approximation algorithm:\n"
    end
    m.bc_started = true
    m.queue_heur = Vector{Vector{Float64}}[]

    # Add lazy cuts callback to solve the conic subproblem, add lazy cuts, and save a heuristic solution if conic solution is best
    function callback_lazy(cb)
        # Save callback reference so can use to adding lazy cuts
        m.model_mip.ext[:cb] = cb

        # Reset cones summary values and calculate outer infeasibility of MIP solution
        reset_cone_summary!(m)
        calc_inf_outer!(m, logs)

        # Solve conic subproblem given integer solution, add lazy cuts to MIP, calculate cut and dual infeasibilities, add solution to heuristic queue vectors if best objective
        process_conic!(m, getvalue(m.x_mip[m.cols_int]), logs)

        # Print cone infeasibilities
        print_inf(m)
    end
    addlazycallback(m.model_mip, callback_lazy)

    # Add heuristic callback to add each feasible solution from the current heuristic queue
    function callback_heur(cb)
        # Take each heuristic solution vector and add as a solution to the MIP
        tic()
        if !isempty(m.queue_heur)
            # Get solution information
            (soln_int, soln_sub, b_sub_int) = pop!(m.queue_heur)

            # Set values of original variables
            for (val, var) in zip(soln_int, m.x_mip[m.cols_int])
                setsolutionvalue(cb, var, val)
            end
            for (val, var) in zip(soln_sub, m.x_mip[m.cols_cont])
                setsolutionvalue(cb, var, val)
            end

            # Set values of MIP-added slacks and helper and disaggregated variables for nonlinear cones
            slck_sub = b_sub_int - m.A_sub_cont * soln_sub
            for n in 1:m.num_cone_nlnr
                add_cone_soln!(m, m.map_spec[n], m.map_dim[n], m.map_vars[n], m.map_isnew[n], slck_sub[m.map_rows_sub[n]], cb)
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
    # TODO gap logic?
    m.gap_rel_opt = abs(m.obj_mip - m.obj_best) / (abs(m.obj_best) + 1e-5)
    logs[:mip_solve] = time() - logs[:mip_solve]
end


#=========================================================
 Conic functions
=========================================================#

# Solve the initial conic relaxation model
function process_relax!(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    tic()

    # Instantiate and solve the conic relaxation model
    model_relax = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_relax, m.c_new, m.A_new, m.b_new, m.cone_con_new, m.cone_var_new)
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
    for n in 1:m.num_cone_nlnr
        add_cone_cuts!(m, m.map_spec[n], m.summary[m.map_spec[n]], m.map_dim[n], m.map_vars[n], m.map_coefs[n], dual_con[m.map_rows[n]])
    end
    logs[:relax_cuts] += toq()

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_relax)
        MathProgBase.freemodel!(model_relax)
    end

    return true
end

# Solve a conic subproblem given some solution to the integer variables
function process_conic!(m::PajaritoConicModel, soln_int::Vector{Float64}, logs::Dict{Symbol,Real})
    tic()

    # Check if integer solution vector is valid
    if any((val -> isnan(val)), soln_int)
        if m.mip_solver_drives
            println("Current integer solution vector has NaN values; terminating Pajarito\n")
            throw(CallbackAbort())
        else
            error("Current integer solution vector has NaN values; terminating Pajarito\n")
        end
    end

    # Calculate new subproblem b vector using fixing values of int vars
    b_sub_int = m.b_sub - m.A_sub_int * soln_int

    # Instantiate and solve the conic model
    model_conic = MathProgBase.ConicModel(m.cont_solver)
    MathProgBase.loadproblem!(model_conic, m.c_sub_cont, m.A_sub_cont, b_sub_int, m.cone_con_sub, m.cone_var_sub)
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
    for n in 1:m.num_cone_nlnr
        add_cone_cuts!(m, m.map_spec[n], m.summary[m.map_spec[n]], m.map_dim[n], m.map_vars[n], m.map_coefs[n], dual_con[m.map_rows_sub[n]])
    end
    logs[:conic_cuts] += toq()

    # If feasible, check if new objective is best
    if status_conic != :Infeasible
        soln_sub = MathProgBase.getsolution(model_conic)
        obj_new = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_sub)

        # If new objective is best, store new objective and solution, and add heuristic solution or warm-start MIP
        if obj_new <= m.obj_best
            m.obj_best = obj_new
            m.soln_best[m.cols_int] = soln_int
            m.soln_best[m.cols_cont] = soln_sub

            if m.mip_solver_drives
                push!(m.queue_heur, [soln_int, soln_sub, b_sub_int])
            else
                # Set values of original variables
                for (val, var) in zip(soln_int, m.x_mip[m.cols_int])
                    setvalue(var, val)
                end
                for (val, var) in zip(soln_sub, m.x_mip[m.cols_cont])
                    setvalue(var, val)
                end

                # Set values of MIP-added slacks and helper and disaggregated variables for nonlinear cones
                slck_sub = b_sub_int - m.A_sub_cont * soln_sub
                for n in 1:m.num_cone_nlnr
                    set_cone_soln!(m, m.map_spec[n], m.map_dim[n], m.map_vars[n], m.map_isnew[n], slck_sub[m.map_rows_sub[n]])
                end
            end
        end
    end

    # Free the conic model
    if applicable(MathProgBase.freemodel!, model_conic)
        MathProgBase.freemodel!(model_conic)
    end
end

# Construct MIP solution on nonlinear cone variables and warm-start
function set_cone_soln!(m::PajaritoConicModel, spec::Symbol, dim::Int, vars::Vector{Vector{JuMP.Variable}}, isnew::Vector{Bool}, slck::Vector{Float64})
    # Set MIP-added slack variable values
    for ind in 1:dim
        if isnew[ind]
            setvalue(vars[1][ind], slck[ind])
        end
    end

    # Set helper variable values and disaggregated variable values
    if (spec == :SOC) && m.disagg_soc && (dim > 2)
        if slck[1] == 0.
            for var in vars[2]
                setvalue(var, 0.)
            end
        else
            for (ind, var) in enumerate(vars[2])
                setvalue(var, (slck[(ind + 1)]^2 / slck[1]))
            end
        end
    end
end

# Construct MIP solution on nonlinear cone variables and add heuristic solution
function add_cone_soln!(m::PajaritoConicModel, spec::Symbol, dim::Int, vars::Vector{Vector{JuMP.Variable}}, isnew::Vector{Bool}, slck::Vector{Float64}, cb)
    # Set MIP-added slack variable values
    for ind in 1:dim
        if isnew[ind]
            setsolutionvalue(cb, vars[1][ind], slck[ind])
        end
    end

    # Set helper variable values and disaggregated variable values
    if (spec == :SOC) && m.disagg_soc && (dim > 2)
        if slck[1] == 0.
            for var in vars[2]
                setsolutionvalue(cb, var, 0.)
            end
        else
            for (ind, var) in enumerate(vars[2])
                setsolutionvalue(cb, var, (slck[(ind + 1)]^2 / slck[1]))
            end
        end
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
            spec = m.map_spec[n]
            soln = m.map_coefs[n] .* getvalue(m.map_vars[n][1])

            if spec == :SOC
                inf_outer = sumabs2(soln[2:end]) - soln[1]^2

            elseif spec == :ExpPrimal
                # TODO consider other case?
                inf_outer = soln[2] * exp(soln[1] / soln[2]) - soln[3]

            elseif spec == :SDP
                inf_outer = -eigmin(make_smat(soln))
            end

            if inf_outer > 0.
                m.summary[spec][:outer_max_n] += 1
                m.summary[spec][:outer_max] = max(inf_outer, m.summary[spec][:outer_max])
            elseif inf_outer < 0.
                m.summary[spec][:outer_min_n] += 1
                m.summary[spec][:outer_min] = max(-inf_outer, m.summary[spec][:outer_min])
            end
        end
        logs[:outer_inf] += toq()
    end
end


#=========================================================
 Cut adding functions
=========================================================#

# Process dual vector and add dual cuts to MIP
function add_cone_cuts!(m::PajaritoConicModel, spec::Symbol, spec_summ::Dict{Symbol,Real}, dim::Int, vars::Vector{Vector{JuMP.Variable}}, coefs::Vector{Float64}, dual::Vector{Float64})
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
    if spec == :SOC
        inf_dual = sumabs2(dual[2:end]) - dual[1]^2

        if (dual[1] > 0.) && (!m.drop_dual_infeas || (inf_dual <= 0.))
            if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
                # Epigraph variable equals norm
                dual = vcat(norm(dual[2:end]), dual[2:end])
            end

            if m.disagg_soc && (length(dual) > 2)
                # Disaggregated cuts
                for ind in 2:dim
                    add_linear_cut!(m, spec_summ, [(coefs[1] * vars[1][1]), vars[2][ind - 1], (coefs[ind] * vars[1][ind])], [(dual[ind] / dual[1])^2, 1., (2 * dual[ind] / dual[1])])
                end
            else
                # Nondisaggregated cuts
                add_linear_cut!(m, spec_summ, (coefs .* vars[1]), dual)
            end
        end

    elseif spec == :ExpPrimal
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
            inf_dual = -dual[1] * exp(dual[2] / dual[1]) - e * dual[3]

            if !m.drop_dual_infeas || (inf_dual <= 0.)
                if ((inf_dual > 0.) && m.proj_dual_infeas) || ((inf_dual < 0.) && m.proj_dual_feas)
                    # Epigraph variable equals LHS
                    dual = vcat(dual[1], dual[2], (-dual[1] * exp(dual[2] / dual[1])) / e)
                end

                add_linear_cut!(m, spec_summ, (coefs .* vars[1]), dual)
            end
        end

    elseif spec == :SDP
        # Get eigendecomposition
        (eigvals_dual, eigvecs_dual) = eig(make_smat(dual))
        inf_dual = -minimum(eigvals_dual)
        n_svec = dim * (dim + 1) / 2

        # If using eigenvector cuts, add SOC or linear eig cuts, else add linear cut
        if m.sdp_eig
            # Get array of (orthonormal) eigenvector columns with significant nonnegative eigenvalues, and sanitize eigenvectors
            Vdual = eigvecs_dual[:, (eigvals_dual .>= m.sdp_tol_eigval)]
            Vdual[abs(Vdual) .< m.sdp_tol_eigvec] = 0.

            if size(Vdual, 2) > 0
                # Cannot add SOC cuts during MIP solve
                # if m._sdp_soc && !m.bc_started
                # TODO broken because of coefs
                #     add_sdp_soc_cuts!(m, spec_summ, reshape(vars[(n_svec + 1):end], dim, dim), coefs, Vdual)
                # else
                    # Add linear cut for each significant eigenvector
                    for jV in 1:size(Vdual, 2)
                        add_linear_cut!(m, spec_summ, (coefs .* vars[1]), make_svec(Vdual[:, jV] * Vdual[:, jV]'))
                    end
                # end
            end

        elseif any(eigvals_dual .>= 0.) && (!m.drop_dual_infeas || (inf_dual <= 0.))
            if (inf_dual > 0.) && m.proj_dual_infeas
                # Project by taking sum of nonnegative eigenvalues times outer products of corresponding eigenvectors
                eigvals_dual[eigvals_dual .<= 0.] = 0.
                dual = make_svec(eigvecs_dual * diagm(eigvals_dual) * eigvecs_dual')
            end

            add_linear_cut!(m, spec_summ, (coefs .* vars[1]), dual)
        end
    end

    # Update dual infeasibility
    if m.log_level > 1
        if inf_dual > 0.
            spec_summ[:dual_max_n] += 1
            spec_summ[:dual_max] = max(inf_dual, spec_summ[:dual_max])
        elseif inf_dual < 0.
            spec_summ[:dual_min_n] += 1
            spec_summ[:dual_min] = max(-inf_dual, spec_summ[:dual_min])
        end
    end
end

# Add a single linear cut and calculate cut infeasibility
function add_linear_cut!(m::PajaritoConicModel, spec_summ::Dict{Symbol,Real}, slcks::Vector{JuMP.AffExpr}, cut::Vector{Float64})
    # Add cut (lazy cut if using MIP driven solve)
    if haskey(m.model_mip.ext, :cb)
        @lazyconstraint(m.model_mip.ext[:cb], dot(cut, slcks) >= 0.)
    else
        @constraint(m.model_mip, dot(cut, slcks) >= 0.)
    end

    # Update cut infeasibility
    if m.log_level > 1
        inf_cut = -dot(cut, getvalue(slcks))
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
# TODO broken
# function add_sdp_soc_cuts!(m::PajaritoConicModel, spec_summ::Dict{Symbol,Real}, vars::Tuple{Vector{JuMP.Variable},Vector{JuMP.Variable},Array{JuMP.Variable,2}}, coefs::Vector{Float64}, Vdual::Array{Float64,2})
#     (nSD, nV) = size(Vdual)
#
#     # For each sanitized eigenvector with significant eigenvalue
#     for jV in 1:nV
#         # Get eigenvector and form rank-1 outer product
#         vj = Vdual[:, jV]
#         vvj = vj * vj'
#
#         # For each diagonal element of SDP
#         for iSD in 1:nSD
#             no_i = vcat(1:(iSD - 1), (iSD + 1):nSD)
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
#             if m.log_level > 1
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
 Algorithm utilities
=========================================================#

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
    logs[:trans_data] = 0.  # Transforming data
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
        for (spec, spec_summ) in summary
            @printf "%10s | %8d | %8d | %8d\n" spec spec_summ[:count] spec_summ[:min_dim] spec_summ[:max_dim]
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
        for (spec, spec_summ) in m.summary
            @printf "%10s | %5d  %8.2e  %5d  %8.2e\n" spec spec_summ[:dual_max_n] spec_summ[:dual_max] spec_summ[:dual_min_n] spec_summ[:dual_min]
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
        for (spec, spec_summ) in m.summary
            @printf "%10s | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e | %5d  %8.2e  %5d  %8.2e\n" spec spec_summ[:outer_max_n] spec_summ[:outer_max] spec_summ[:outer_min_n] spec_summ[:outer_min] spec_summ[:dual_max_n] spec_summ[:dual_max] spec_summ[:dual_min_n] spec_summ[:dual_min] spec_summ[:cut_max_n] spec_summ[:cut_max] spec_summ[:cut_min_n] spec_summ[:cut_min]
        end

        @printf "\n"
        flush(STDOUT)
    end
end

# Print objective gap information
function print_gap(m::PajaritoConicModel, logs::Dict{Symbol,Real})
    if m.log_level > 0
        if (logs[:n_conic] == 1) || (m.log_level > 1)
            @printf "\n%-4s | %-14s | %-14s | %-11s | %-11s\n" "Iter" "Best obj" "OA obj" "Rel gap" "Time (s)"
        end

        if m.gap_rel_opt < 1000
            @printf "%4d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" logs[:n_conic] m.obj_best m.obj_mip m.gap_rel_opt (time() - logs[:oa_alg])
        elseif isnan(m.gap_rel_opt)
            @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" logs[:n_conic] m.obj_best m.obj_mip "Inf" (time() - logs[:oa_alg])
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
    m.log_level > 0 || return
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
            @printf " -- Transform data      = %14.2e\n" logs[:trans_data]
            @printf " -- Create MIP data     = %14.2e\n" logs[:mip_data]
            @printf " -- Create conic data   = %14.2e\n" logs[:conic_data]
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
