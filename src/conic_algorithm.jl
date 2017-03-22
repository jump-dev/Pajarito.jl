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


TODO features
- implement warm-starting: use set_best_soln!
- enable querying logs information etc

=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # Solver parameters
    log_level::Int              # Verbosity flag: 0 for minimal information, 1 for basic solve statistics, 2 for iteration information, 3 for cone information
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
    solve_subp::Bool            # (Conic only) Solve the continuous conic subproblems to add subproblem cuts
    dualize_relax::Bool         # (Conic only) Solve the conic dual of the continuous conic relaxation
    dualize_subp::Bool          # (Conic only) Solve the conic duals of the continuous conic subproblems

    soc_disagg::Bool            # (Conic only) Disaggregate SOC cones in the MIP only
    soc_abslift::Bool           # (Conic only) Use SOC absolute value lifting in the MIP only
    soc_in_mip::Bool            # (Conic only) Use SOC cones in the MIP outer approximation model (if MIP solver supports MISOCP)
    sdp_eig::Bool               # (Conic SDP only) Use SDP eigenvector-derived cuts
    sdp_soc::Bool               # (Conic SDP only) Use SDP eigenvector SOC cuts (if MIP solver supports MISOCP; except during MIP-driven solve)
    init_soc_one::Bool          # (Conic only) Start with disaggregated L_1 outer approximation cuts for SOCs
    init_soc_inf::Bool          # (Conic only) Start with disaggregated L_inf outer approximation cuts for SOCs
    init_exp::Bool              # (Conic Exp only) Start with several outer approximation cuts on the exponential cones
    init_sdp_lin::Bool          # (Conic SDP only) Use SDP initial linear cuts
    init_sdp_soc::Bool          # (Conic SDP only) Use SDP initial SOC cuts (if MIP solver supports MISOCP)

    scale_subp_cuts::Bool       # (Conic only) Use scaling for subproblem cuts based on subproblem status
    scale_factor::Float64       # (Conic only) Multiplicative factor for scaled subproblem cuts (cuts are scaled by scale_factor*tol_prim_infeas/rel_gap)
    viol_cuts_only::Bool        # (Conic only) Only add cuts that are violated by the current MIP solution (may be useful for MSD algorithm where many cuts are added)
    prim_cuts_only::Bool        # (Conic only) Do not add subproblem cuts
    prim_cuts_always::Bool      # (Conic only) Add primal cuts at each iteration or in each lazy callback
    prim_cuts_assist::Bool      # (Conic only) Add primal cuts only when integer solutions are repeating or when conic solver fails

    tol_zero::Float64           # (Conic only) Tolerance for small epsilons as zeros
    tol_prim_infeas::Float64    # (Conic only) Tolerance level for cone outer infeasibilities for primal cut adding functions (should be at least 1e-5)

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
    r_idx_exp_subp::Vector{Int} # Row index of r variable in ExpPrimals in subproblems
    s_idx_exp_subp::Vector{Int} # Row index of s variable in ExpPrimals in subproblems
    t_idx_exp_subp::Vector{Int} # Row index of t variable in ExpPrimals in subproblems
    r_exp::Vector{JuMP.AffExpr} # r variable in ExpPrimals
    s_exp::Vector{JuMP.AffExpr} # s variable in ExpPrimals
    t_exp::Vector{JuMP.AffExpr} # t variable in ExpPrimals

    # SDP data
    num_sdp::Int                # Number of SDP cones
    v_idxs_sdp_subp::Vector{Vector{Int}} # Row indices of svec v variables in SDPs in subproblem
    smat_sdp::Vector{Symmetric{Float64,Array{Float64,2}}} # Preallocated array for smat space values
    V_sdp::Vector{Array{JuMP.AffExpr,2}} # smat V variables in SDPs

    # Miscellaneous for algorithms
    new_scale_factor::Float64   # Calculated value for subproblem cuts scaling
    update_conicsub::Bool       # Indicates whether to use setbvec! to update an existing conic subproblem model
    model_conic::MathProgBase.AbstractConicModel # Conic subproblem model: persists when the conic solver implements MathProgBase.setbvec!
    oa_started::Bool            # Indicator for Iterative or MIP-solver-driven algorithms started
    cache_dual::Dict{Vector{Float64},Vector{Float64}} # Set of integer solution subvectors already seen
    new_incumb::Bool            # Indicates whether a new incumbent solution from the conic solver is waiting to be added as warm-start or heuristic
    cb_heur                     # Heuristic callback reference (MIP-driven only)
    cb_lazy                     # Lazy callback reference (MIP-driven only)

    # Solution and bound information
    is_best_conic::Bool         # Indicates best feasible came from conic solver solution, otherwise MIP solver solution
    mip_obj::Float64            # Latest MIP (outer approx) objective value
    best_obj::Float64           # Best feasible objective value
    best_int::Vector{Float64}   # Best feasible integer solution
    best_cont::Vector{Float64}  # Best feasible continuous solution
    gap_rel_opt::Float64        # Relative optimality gap = |mip_obj - best_obj|/|best_obj|
    final_soln::Vector{Float64} # Final solution on original variables

    # Logging information and status
    logs::Dict{Symbol,Any}      # Logging information
    status::Symbol              # Current Pajarito status

    # Model constructor
    function PajaritoConicModel(log_level, timeout, rel_gap, mip_solver_drives, mip_solver, mip_subopt_solver, mip_subopt_count, round_mip_sols, pass_mip_sols, cont_solver, solve_relax, solve_subp, dualize_relax, dualize_subp, soc_disagg, soc_abslift, soc_in_mip, sdp_eig, sdp_soc, init_soc_one, init_soc_inf, init_exp, init_sdp_lin, init_sdp_soc, scale_subp_cuts, scale_factor, viol_cuts_only, prim_cuts_only, prim_cuts_always, prim_cuts_assist, tol_zero, tol_prim_infeas)
        m = new()

        m.log_level = log_level
        m.mip_solver_drives = mip_solver_drives
        m.solve_relax = solve_relax
        m.solve_subp = solve_subp
        m.dualize_relax = dualize_relax
        m.dualize_subp = dualize_subp
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
        m.scale_factor = scale_factor
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
        error("Must call setvartype! immediately after loadproblem!\n")
    end
    if length(var_types) != m.num_var_orig
        error("Variable types vector length ($(length(var_types))) does not match number of variables ($(m.num_var_orig))\n")
    end
    if any((var_type -> (var_type != :Bin) && (var_type != :Int) && (var_type != :Cont)), var_types)
        error("Some variable types are not :Bin, :Int, :Cont\n")
    end
    if !any((var_type -> (var_type == :Bin) || (var_type == :Int)), var_types)
        error("No variable types are :Bin or :Int; use the continuous conic solver directly if your problem is continuous\n")
    end

    m.var_types = var_types
end

# Solve, given the initial conic model data and the variable types vector and possibly a warm-start vector
function MathProgBase.optimize!(m::PajaritoConicModel)
    if m.status != :Loaded
        error("Must call optimize! after setvartype! and loadproblem!\n")
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
    (c_new, A_new, b_new, cone_con_new, cone_var_new, keep_cols, var_types_new, cols_cont, cols_int) = transform_data(copy(m.c_orig), copy(m.A_orig), copy(m.b_orig), deepcopy(m.cone_con_orig), deepcopy(m.cone_var_orig), copy(m.var_types), m.solve_relax)
    m.logs[:data_trans] += toq()
    if m.log_level > 1
        @printf "%.2fs\n" m.logs[:data_trans]
    end

    if m.solve_subp
        # Create conic subproblem data
        if m.log_level > 1
            @printf "\nCreating conic model data..."
        end
        tic()
        map_rows_subp = create_conicsub_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, cols_cont, cols_int)
        m.logs[:data_conic] += toq()
        if m.log_level > 1
            @printf "%.2fs\n" m.logs[:data_conic]
        end
    else
        map_rows_subp = zeros(Int, length(b_new))
        m.c_sub_cont = c_new[cols_cont]
        m.c_sub_int = c_new[cols_int]
    end

    # Create MIP model
    if m.log_level > 1
        @printf "\nCreating MIP model..."
    end
    tic()
    (v_idxs_soc_relx, r_idx_exp_relx, s_idx_exp_relx, v_idxs_sdp_relx) = create_mip_data!(m, c_new, A_new, b_new, cone_con_new, cone_var_new, var_types_new, map_rows_subp, cols_cont, cols_int)
    m.logs[:data_mip] += toq()
    if m.log_level > 1
        @printf "%.2fs\n" m.logs[:data_mip]
    end

    # Calculate subproblem cuts scaling factor
    m.new_scale_factor = m.scale_factor*m.tol_prim_infeas/m.rel_gap*(m.num_soc + m.num_exp + m.num_sdp)

    if m.solve_relax
        # Solve relaxed conic problem, proceed with algorithm if optimal or suboptimal, else finish
        if m.log_level > 1
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
        if m.log_level > 1
            @printf "%.2fs\n" m.logs[:relax_solve]
        end

        status_relax = MathProgBase.status(model_relax)
        obj_relax = MathProgBase.getobjval(model_relax)

        if status_relax == :Infeasible
            if m.log_level > 0
                println("Initial conic relaxation status was $status_relax")
            end
            m.status = :Infeasible
        elseif status_relax == :Unbounded
            if m.log_level > 0
                println("Initial conic relaxation status was $status_relax")
            end
            m.status = :Unbounded
        elseif (status_relax != :Optimal) && (status_relax != :Suboptimal)
            warn("Conic solver failure on initial relaxation: returned status $status_relax\n")
            m.status = :CutsFailure
        elseif isnan(obj_relax)
            warn("Conic solver had objective value $obj_relax: returned status $status_relax\n")
            m.status = :CutsFailure
        else
            if m.log_level > 2
                @printf " - Relaxation status    = %14s\n" status_relax
                @printf " - Relaxation objective = %14.6f\n" obj_relax
            end

            # Get dual and check for NaNs
            dual_conic = MathProgBase.getdual(model_relax)
            if !isempty(dual_conic) && !any(isnan, dual_conic)
                # Optionally scale dual
                if m.scale_subp_cuts
                    # Rescale by number of cones / absval of full conic objective
                    scale!(dual_conic, m.new_scale_factor/(abs(obj_relax) + 1e-5))
                end

                # Add relaxation cuts
                tic()
                add_subp_cuts!(m, dual_conic, v_idxs_soc_relx, r_idx_exp_relx, s_idx_exp_relx, v_idxs_sdp_relx)
                m.logs[:relax_cuts] += toq()
            end
        end

        # Free the conic model
        if applicable(MathProgBase.freemodel!, model_relax)
            MathProgBase.freemodel!(model_relax)
        end
    end

    # Finish if exceeded timeout option
    if (time() - m.logs[:total]) > m.timeout
        m.status = :UserLimit
    end

    if (m.status != :UserLimit) && (m.status != :Infeasible) && (m.status != :Unbounded) && (m.prim_cuts_assist || (m.status != :CutsFailure))
        if m.solve_subp
            if m.log_level > 2
                @printf "\nCreating conic subproblem model..."
            end
            tic()
            if m.dualize_subp
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
            if m.log_level > 2
                @printf "%.2fs\n" toq()
            end
        end

        # Initialize and begin iterative or MIP-solver-driven algorithm
        m.oa_started = true
        m.new_incumb = false
        m.cache_dual = Dict{Vector{Float64},Vector{Float64}}()

        if m.mip_solver_drives
            if m.log_level > 1
                @printf "\nStarting MIP-solver-driven outer approximation algorithm\n"
            end
            solve_mip_driven!(m)
        else
            if m.log_level > 1
                @printf "\nStarting iterative outer approximation algorithm\n"
            end
            solve_iterative!(m)
        end

        if m.best_obj < Inf
            # Have a best feasible solution, update final solution on original variables
            soln_new = zeros(length(c_new))
            soln_new[cols_int] = m.best_int
            soln_new[cols_cont] = m.best_cont
            m.final_soln = zeros(m.num_var_orig)
            m.final_soln[keep_cols] = soln_new[1:length(keep_cols)]
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
    if m.log_level > 1
        @printf "\nCone dimensions summary:"
        @printf "\n%-16s | %-9s | %-9s | %-9s\n" "Cone" "Count" "Min dim." "Max dim."
        if num_soc > 0
            @printf "%16s | %9d | %9d | %9d\n" "Second order" num_soc min_soc max_soc
        end
        if num_rot > 0
            @printf "%16s | %9d | %9d | %9d\n" "Rotated S.O." num_rot min_rot max_rot
        end
        if num_exp > 0
            @printf "%16s | %9d | %9d | %9d\n" "Primal expon." num_exp 3 3
        end
        if num_sdp > 0
            min_side = round(Int, sqrt(1/4+2*min_sdp)-1/2)
            max_side = round(Int, sqrt(1/4+2*max_sdp)-1/2)
            @printf "%16s | %9d | %7s^2 | %7s^2\n" "Pos. semidef." num_sdp min_side max_side
        end
        flush(STDOUT)
    end
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
    # (y,z,x) in RSOC <=> (1/sqrt2*(y+z),1/sqrt2*w,x) in SOC, y >= 0, z >= 0, w >= -y+z, w >= -z+y
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

        # Add new variable cone for w
        num_var_new += 1
        push!(cone_var_new, (:NonNeg, [num_var_new]))
        push!(c_new, 0.)
        push!(var_types_new, :Cont)

        # Add new constraint cone for w+y-z >= 0
        num_con_new += 1
        push!(cone_con_new, (:NonNeg, [num_con_new]))
        append!(A_I, fill(num_con_new, (length(inds_1) + length(inds_2) + 1)))
        push!(A_J, num_var_new)
        append!(A_J, A_J[inds_1])
        append!(A_J, A_J[inds_2])
        push!(A_V, -1.)
        append!(A_V, A_V[inds_1])
        append!(A_V, -A_V[inds_2])
        push!(b_new, (b_new[rows[1]] - b_new[rows[2]]))

        # Add new constraint cone for w-y+z >= 0
        num_con_new += 1
        push!(cone_con_new, (:NonNeg, [num_con_new]))
        append!(A_I, fill(num_con_new, (length(inds_1) + length(inds_2) + 1)))
        push!(A_J, num_var_new)
        append!(A_J, A_J[inds_1])
        append!(A_J, A_J[inds_2])
        push!(A_V, -1.)
        append!(A_V, -A_V[inds_1])
        append!(A_V, A_V[inds_2])
        push!(b_new, (-b_new[rows[1]] + b_new[rows[2]]))

        # Use old constraint cone SOCRotated for (y+z,w,sqrt2*x) in SOC
        # Set up index 1: y -> y+z
        append!(A_I, fill(rows[1], length(inds_2)))
        append!(A_J, A_J[inds_2])
        append!(A_V, A_V[inds_2])
        b_new[rows[1]] += b_new[rows[2]]

        # Set up index 2: z -> w
        for ind in inds_2
            A_V[ind] = 0.
        end
        push!(A_I, rows[2])
        push!(A_J, num_var_new)
        push!(A_V, -1.)
        b_new[rows[2]] = 0.

        # Multiply x by sqrt(2)
        b_new[rows[3:end]] .*= sqrt2
        for i in rows[3:end]
            for ind in row_to_nzind[i]
                A_V[ind] *= sqrt2
            end
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

    # Collect indices of continuous and integer variables
    cols_cont = find(vt -> (vt == :Cont), var_types_new)
    cols_int = find(vt -> (vt != :Cont), var_types_new)

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
    map_rows_subp = Vector{Int}(num_con_new)

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
    v_idxs_sdp_relx = Vector{Vector{Int}}(m.num_sdp)
    v_idxs_sdp_subp = Vector{Vector{Int}}(m.num_sdp)
    smat_sdp = Vector{Symmetric{Float64,Array{Float64,2}}}(m.num_sdp)
    V_sdp = Vector{Array{JuMP.AffExpr,2}}(m.num_sdp)

    # Add constraint cones to MIP; if linear, add directly, else create slacks if necessary
    n_soc = 0
    n_exp = 0
    n_sdp = 0

    @expression(model_mip, lhs_expr, b_new - A_new * x_all)

    for (spec, rows) in cone_con_new
        if spec == :NonNeg
            @constraint(model_mip, lhs_expr[rows] .>= 0)
        elseif spec == :NonPos
            @constraint(model_mip, lhs_expr[rows] .<= 0.)
        elseif spec == :Zero
            @constraint(model_mip, lhs_expr[rows] .== 0.)
        elseif spec == :SOC
            # Set up a SOC
            # (t,v) in SOC <-> t >= norm(v)
            n_soc += 1

            if m.soc_in_mip
                # If putting SOCs in the MIP directly, don't need to use other SOC infrastructure
                @constraint(model_mip, lhs_expr[rows[1]] >= norm(lhs_expr[rows[2:end]]))

                v_idxs_soc_relx[n_soc] = Int[]
                t_idx_soc_subp[n_soc] = 0
                v_idxs_soc_subp[n_soc] = Int[]
                continue
            end

            v_idxs = rows[2:end]
            dim = length(v_idxs)
            v_idxs_soc_relx[n_soc] = v_idxs
            t_idx_soc_subp[n_soc] = map_rows_subp[rows[1]]
            v_idxs_soc_subp[n_soc] = map_rows_subp[v_idxs]

            t_soc[n_soc] = t = lhs_expr[rows[1]]
            v_soc[n_soc] = v = lhs_expr[v_idxs]

            if m.soc_disagg
                # Add disaggregated SOC variables d_j
                # 2*d_j >= v_j^2/t, all j
                d = @variable(model_mip, [j in 1:dim], lowerbound=0)
                for j in 1:dim
                    setname(d[j], "d$(j)_soc$(n_soc)")
                end

                # Add disaggregated SOC constraint
                # t >= sum(2*d_j)
                # Scale by 2
                @constraint(model_mip, 2*(t - 2*sum(d)) >= 0)
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

                # Add absolute value SOC constraints
                # a_j >= v_j, a_j >= -v_j
                # Scale by 2
                for j in 1:dim
                    @constraint(model_mip, 2*(a[j] - v[j]) >= 0)
                    @constraint(model_mip, 2*(a[j] + v[j]) >= 0)
                end
            else
                a = Vector{JuMP.Variable}()
            end

            d_soc[n_soc] = d
            a_soc[n_soc] = a

            # Set bounds
            @constraint(model_mip, t >= 0)

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
                elseif m.soc_abslift
                    # Using absvalue lifting only
                    # t >= 1/sqrt(dim)*sum(a_j)
                    # Scale by 2
                    @constraint(model_mip, 2*(t - 1/sqrt(dim)*sum(a)) >= 0)
                else
                    Base.warn_once("Cannot use initial SOC L_1 constraints if not using SOC disaggregation or SOC absvalue lifting\n")
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
        elseif spec == :ExpPrimal
            # Set up a ExpPrimal cone
            # (r,s,t) in ExpPrimal <-> t >= s*exp(r/s)
            n_exp += 1
            r_idx_exp_relx[n_exp] = rows[1]
            s_idx_exp_relx[n_exp] = rows[2]
            r_idx_exp_subp[n_exp] = map_rows_subp[rows[1]]
            s_idx_exp_subp[n_exp] = map_rows_subp[rows[2]]
            t_idx_exp_subp[n_exp] = map_rows_subp[rows[3]]

            r_exp[n_exp] = r = lhs_expr[rows[1]]
            s_exp[n_exp] = s = lhs_expr[rows[2]]
            t_exp[n_exp] = t = lhs_expr[rows[3]]

            # Set bounds
            @constraint(model_mip, s >= 0)
            @constraint(model_mip, t >= 0)

            if m.init_exp
                # Add initial exp cuts using dual exp cone linearizations
                # (u,v,w) in ExpDual <-> exp(1)*w >= -u*exp(v/u), w >= 0, u < 0
                # at u = -1; v = -1, -1/2, -1/5, 0, 1/5, 1/2, 1; z = exp(-v-1)
                for v in [-1., -0.5, -0.2, 0., 0.2, 0.5, 1.]
                    @constraint(model_mip, -r + v*s + exp(-v-1)*t >= 0)
                end
            end
        elseif spec == :SDP
            # Set up a PSD cone
            # V_svec in SDP <-> V_smat >= 0
            n_sdp += 1
            dim = round(Int, sqrt(1/4+2*length(rows))-1/2) # smat space side dimension

            v_idxs_sdp_relx[n_sdp] = rows
            v_idxs_sdp_subp[n_sdp] = map_rows_subp[rows]

            smat_sdp[n_sdp] = Symmetric(zeros(dim, dim))

            v = lhs_expr[rows]
            V_sdp[n_sdp] = V = Array{JuMP.AffExpr,2}(dim, dim)

            # Set up smat arrays and set bounds
            kSD = 1
            for jSD in 1:dim, iSD in jSD:dim
                if jSD == iSD
                    @constraint(model_mip, v[kSD] >= 0)
                    V[iSD, jSD] = v[kSD]
                else
                    V[iSD, jSD] = V[jSD, iSD] = sqrt2inv*v[kSD]
                end
                kSD += 1
            end

            # Add initial (linear or SOC) SDP outer approximation cuts
            if m.init_sdp_soc
                for jSD in 1:dim, iSD in (jSD+1):dim
                    # Add initial SOC cut for off-diagonal element to enforce 2x2 principal submatrix PSDness
                    # yz >= ||x||^2, y,z >= 0 <==> norm2(2x, y-z) <= y + z
                    @constraint(model_mip, V[iSD, iSD] + V[jSD, jSD] - norm(JuMP.AffExpr[2*V[iSD, jSD], V[iSD, iSD] - V[jSD, jSD]]) >= 0)
                end
            elseif m.init_sdp_lin
                for jSD in 1:dim, iSD in (jSD+1):dim
                    # Add initial linear cuts based on linearization of 3-dim rotated SOCs that enforce 2x2 principal submatrix PSDness (essentially the dual of SDSOS)
                    # 2|m_ij| <= m_ii + m_jj, where m_kk is scaled by sqrt2 in smat space
                    @constraint(model_mip, V[iSD, iSD] + V[jSD, jSD] - 2*V[iSD, jSD] >= 0)
                    @constraint(model_mip, V[iSD, iSD] + V[jSD, jSD] + 2*V[iSD, jSD] >= 0)
                end
            end

            if m.sdp_soc && m.mip_solver_drives
                Base.warn_once("SOC cuts for SDP cones cannot be added in callbacks in the MIP-solver-driven algorithm\n")
            end
        end
    end

    # Store MIP data
    m.model_mip = model_mip
    m.x_int = x_all[cols_int]
    m.x_cont = x_all[cols_cont]
    # @show model_mip

    if m.soc_in_mip
        m.num_soc = 0
    end
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

    m.v_idxs_sdp_subp = v_idxs_sdp_subp
    m.smat_sdp = smat_sdp
    m.V_sdp = V_sdp

    return (v_idxs_soc_relx, r_idx_exp_relx, s_idx_exp_relx, v_idxs_sdp_relx)
end


#=========================================================
 Algorithms
=========================================================#

# Solve the MIP model using iterative outer approximation algorithm
function solve_iterative!(m)
    count_subopt = 0

    while true
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

        # Solve MIP
        tic()
        status_mip = solve(m.model_mip, suppress_warnings=true)
        m.logs[:mip_solve] += toq()
        m.logs[:n_iter] += 1

        if (status_mip == :Infeasible) || (status_mip == :InfeasibleOrUnbounded)
            # Stop if infeasible
            m.status = :Infeasible
            break
        elseif status_mip == :Unbounded
            # Stop if unbounded (initial conic relax solve should detect this)
            if !m.solve_relax
                warn("MIP solver returned status $status_mip but the conic relaxation problem was not solved (set solve_relax = true)\n")
            else
                warn("MIP solver returned status $status_mip, which could indicate a problem with the conic relaxation solve\n")
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
        (is_repeat, is_viol_subp) = solve_subp_add_subp_cuts!(m)

        # Calculate relative outer approximation gap, finish if satisfy optimality gap condition
        m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
        if m.gap_rel_opt < m.rel_gap
            print_gap(m)
            m.status = :Optimal
            break
        end

        if !is_viol_subp || m.prim_cuts_always
            # No violated subproblem cuts added, or always adding primal cuts
            # Check feasibility and add primal cuts if primal cuts for convergaence assistance
            (is_infeas, is_viol_prim) = check_feas_add_prim_cuts!(m, m.prim_cuts_assist)

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
                    m.is_best_conic = false

                    # Calculate relative outer approximation gap, finish if satisfy optimality gap condition
                    m.gap_rel_opt = (m.best_obj - m.mip_obj) / (abs(m.best_obj) + 1e-5)
                    if m.gap_rel_opt < m.rel_gap
                        print_gap(m)
                        m.status = :Optimal
                        break
                    end
                end
            end

            if is_repeat && !is_viol_prim
                # Integer solution has repeated and no violated cuts were added
                if count_subopt == 0
                    # Solve was optimal solve, so nothing more we can do
                    if m.prim_cuts_assist
                        warn("No violated cuts were added on repeated integer solution (this should not happen: please submit an issue)\n")
                    else
                        warn("No violated cuts were added on repeated integer solution (try using prim_cuts_assist = true)\n")
                    end
                    m.status = :CutsFailure
                    break
                end

                # Try solving next MIP to optimality, if that doesn't help then we will fail next iteration
                warn("Integer solution has repeated and no violated cuts were added: solving next MIP to optimality\n")
                count_subopt = m.mip_subopt_count
            end
        end

        print_gap(m)

        # Finish if exceeded timeout option
        if (time() - m.logs[:total]) > m.timeout
            m.status = :UserLimit
            break
        end

        if m.pass_mip_sols && isfinite(m.best_obj)
            # Give the best feasible solution to the MIP as a warm-start
            m.logs[:n_add] += 1
            set_best_soln!(m, m.best_int, m.best_cont)
        else
            # For MIP solvers that accept warm starts without checking feasibility, set all variables to NaN
            for var in m.x_int
                setvalue(var, NaN)
            end
            for var in m.x_cont
                setvalue(var, NaN)
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

        # Solve new conic subproblem, update incumbent solution if feasible
        (is_repeat, is_viol_subp) = solve_subp_add_subp_cuts!(m)

        # Finish if any violated subproblem cuts were added and not using primal cuts always
        if is_viol_subp && !m.prim_cuts_always
            return
        end

        # Check feasibility of current solution, try to add violated primal cuts if using primal cuts for convergence assistance
        (is_infeas, is_viol_prim) = check_feas_add_prim_cuts!(m, m.prim_cuts_assist)

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

    if (status_mip == :Infeasible) || (status_mip == :InfeasibleOrUnbounded)
        m.status = :Infeasible
        return
    elseif status_mip == :Unbounded
        # Stop if unbounded (initial conic relax solve should detect this)
        if !m.solve_relax
            warn("MIP solver returned status $status_mip but the conic relaxation problem was not solved (set solve_relax = true)\n")
        else
            warn("MIP solver returned status $status_mip, which could indicate a problem with the conic relaxation solve\n")
        end
        m.status = :CutsFailure
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
        if m.soc_disagg
            # Set disaggregated SOC variable values
            if getvalue(m.t_soc[n]) < 1e-7
                # If epigraph values is small, set d to 0
                if m.mip_solver_drives && m.oa_started
                    for j in 1:length(m.d_soc[n])
                        setsolutionvalue(m.cb_heur, m.d_soc[n][j], 0)
                    end
                else
                    for j in 1:length(m.d_soc[n])
                        setvalue(m.d_soc[n][j], 0)
                    end
                end
            else
                # Calculate and set d
                if m.mip_solver_drives && m.oa_started
                    for j in 1:length(m.d_soc[n])
                        setsolutionvalue(m.cb_heur, m.d_soc[n][j], getvalue(m.v_soc[n][j])^2/(2.*getvalue(m.t_soc[n])))
                    end
                else
                    for j in 1:length(m.d_soc[n])
                        setvalue(m.d_soc[n][j], getvalue(m.v_soc[n][j])^2/(2.*getvalue(m.t_soc[n])))
                    end
                end
            end
        end

        if m.soc_abslift
            # Set absval lifted variable values
            if m.mip_solver_drives && m.oa_started
                for j in 1:length(m.a_soc[n])
                    setsolutionvalue(m.cb_heur, m.a_soc[n][j], abs(getvalue(m.v_soc[n][j])))
                end
            else
                for j in 1:length(m.a_soc[n])
                    setvalue(m.a_soc[n][j], abs(getvalue(m.v_soc[n][j])))
                end
            end
        end
    end
end

# Transform svec vector into symmetric smat matrix
function make_smat!(smat::Symmetric{Float64,Array{Float64,2}}, svec::Vector{Float64})
    # smat is uplo U Symmetric
    dim = size(smat, 1)
    kSD = 1
    for iSD in 1:dim, jSD in iSD:dim
        if iSD == jSD
            smat.data[iSD, jSD] = svec[kSD]
        else
            smat.data[iSD, jSD] = sqrt2inv*svec[kSD]
        end
        kSD += 1
    end
    return smat
end


#=========================================================
 K^* cuts functions
=========================================================#

# Solve the subproblem for the current integer solution, add new incumbent conic solution if feasible and best, add K* cuts from subproblem dual solution
function solve_subp_add_subp_cuts!(m)
    # Get current integer solution
    soln_int = getvalue(m.x_int)
    if m.round_mip_sols
        # Round the integer values
        soln_int = map!(round, soln_int)
    end

    if haskey(m.cache_dual, soln_int)
        # Integer solution has been seen before, cannot get new subproblem cuts
        m.logs[:n_repeat] += 1

        if !m.mip_solver_drives || m.prim_cuts_only || !m.solve_subp
            # Nothing to do if using iterative, or if not using subproblem cuts
            return (true, false)
        else
            # In MSD, re-add subproblem cuts from existing conic dual
            dual_conic = m.cache_dual[soln_int]
            if isempty(dual_conic)
                # Don't have a conic dual due to conic failure, nothing to do
                return (true, false)
            end
        end
    elseif !m.solve_subp
        return (false, false)
    else
        # Integer solution is new, save it
        m.cache_dual[soln_int] = Float64[]

        # Calculate new b vector from integer solution and solve conic subproblem model
        b_sub_int = m.b_sub - m.A_sub_int*soln_int
        (status_conic, soln_conic, dual_conic) = solve_subp!(m, b_sub_int)

        # Determine cut scaling factors and check if have new feasible incumbent solution
        if (status_conic == :Infeasible) && !isempty(dual_conic)
            # Subproblem infeasible: first check infeasible ray has negative value
            ray_value = vecdot(dual_conic, b_sub_int)
            if ray_value < -m.tol_zero
                if m.scale_subp_cuts
                    # Rescale by number of cones / value of ray
                    scale!(dual_conic, m.new_scale_factor/ray_value)
                end
            else
                warn("Conic solver failure: returned status $status_conic with empty solution and nonempty dual, but b'y is not sufficiently negative for infeasible ray y (this should not happen: please submit an issue)\n")
                return (false, false)
            end
        elseif (status_conic == :Optimal) && !isempty(dual_conic)
            # Subproblem feasible: first calculate full objective value
            obj_full = dot(m.c_sub_int, soln_int) + dot(m.c_sub_cont, soln_conic)

            if m.scale_subp_cuts
                # Rescale by number of cones / abs(objective + 1e-5)
                scale!(dual_conic, m.new_scale_factor/(abs(obj_full) + 1e-5))
            end

            m.logs[:n_feas_conic] += 1
            if obj_full < m.best_obj
                # Conic solver solution is a new incumbent
                m.best_obj = obj_full
                m.best_int = soln_int
                m.best_cont = soln_conic
                m.new_incumb = true
                m.is_best_conic = true
            end
        elseif !isempty(dual_conic)
            # We have a dual but don't know the status, so we can't use subproblem scaling
            if m.scale_subp_cuts
                # Rescale by number of cones
                scale!(dual_conic, m.new_scale_factor)
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

        # In MSD, save the dual so can re-add subproblem cuts later
        if m.mip_solver_drives
            m.cache_dual[soln_int] = dual_conic
        end
    end

    tic()
    is_viol_subp = add_subp_cuts!(m, dual_conic, m.v_idxs_soc_subp, m.r_idx_exp_subp, m.s_idx_exp_subp, m.v_idxs_sdp_subp)
    m.logs[:subp_cuts] += toq()

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
        if m.dualize_subp
            solver_conicsub = ConicDualWrapper(conicsolver=m.cont_solver)
        else
            solver_conicsub = m.cont_solver
        end

        m.model_conic = MathProgBase.ConicModel(solver_conicsub)
        MathProgBase.loadproblem!(m.model_conic, m.c_sub_cont, m.A_sub_cont, b_sub_int, m.cone_con_sub, m.cone_var_sub)
    end

    MathProgBase.optimize!(m.model_conic)
    m.logs[:n_conic] += 1
    m.logs[:subp_solve] += toq()

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

    if status_conic == :Optimal
        soln_conic = MathProgBase.getsolution(m.model_conic)
    else
        # Try to get a solution
        try
            soln_conic = MathProgBase.getsolution(m.model_conic)
        catch
            soln_conic = Float64[]
        end
    end
    if any(isnan, soln_conic)
        soln_conic = Float64[]
    end

    if (status_conic == :Optimal) || (status_conic == :Infeasible)
        dual_conic = MathProgBase.getdual(m.model_conic)
    else
        # Try to get a dual
        try
            dual_conic = MathProgBase.getdual(m.model_conic)
        catch
            dual_conic = Float64[]
        end
    end
    if any(isnan, dual_conic)
        dual_conic = Float64[]
    end

    # Free the conic model if not saving it
    if !m.update_conicsub && applicable(MathProgBase.freemodel!, m.model_conic)
        MathProgBase.freemodel!(m.model_conic)
    end

    return (status_conic, soln_conic, dual_conic)
end

# Add K* cuts from subproblem dual solution
function add_subp_cuts!(m, dual_conic, v_idxs_soc, r_idx_exp, s_idx_exp, v_idxs_sdp)
    is_viol_subp = false

    for n in 1:m.num_soc
        if add_cut_soc!(m, m.t_soc[n], m.v_soc[n], m.d_soc[n], m.a_soc[n], dual_conic[v_idxs_soc[n]])
            is_viol_subp = true
        end
    end

    for n in 1:m.num_exp
        if add_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], dual_conic[r_idx_exp[n]], dual_conic[s_idx_exp[n]])
            is_viol_subp = true
        end
    end

    for n in 1:m.num_sdp
        # Dual is sum_{j: lambda_j > 0} lamda_j V_j V_j'
        v_dual = dual_conic[v_idxs_sdp[n]]
        V_eig = eigfact!(make_smat!(m.smat_sdp[n], v_dual), sqrt(m.tol_zero), Inf)
        if add_cut_sdp!(m, m.V_sdp[n], V_eig[:vectors] * Diagonal(sqrt.(V_eig[:values])))
            is_viol_subp = true
        end
    end

    return is_viol_subp
end

# Check cone infeasibilities of current solution, optionally add K* cuts from current solution for infeasible cones
function check_feas_add_prim_cuts!(m, add_cuts::Bool)
    tic()
    is_infeas = false
    is_viol_prim = false

    for n in 1:m.num_soc
        # Get cone current solution, check infeasibility
        v_vals = getvalue(m.v_soc[n])
        if (vecnorm(v_vals) - getvalue(m.t_soc[n])) < m.tol_prim_infeas
            continue
        end
        is_infeas = true
        if !add_cuts
            continue
        end

        # Dual is (1, -1/norm(v)*v)
        if add_cut_soc!(m, m.t_soc[n], m.v_soc[n], m.d_soc[n], m.a_soc[n], -1/vecnorm(v_vals)*v_vals)
            is_viol_prim = true
        end
    end

    for n in 1:m.num_exp
        r_val = getvalue(m.r_exp[n])
        s_val = getvalue(m.s_exp[n])
        if (s_val*exp(r_val/s_val) - getvalue(m.t_exp[n])) < m.tol_prim_infeas
            continue
        end
        is_infeas = true
        if !add_cuts
            continue
        end

        # Dual is (-exp(r/s), exp(r/s)(r/s-1), 1)
        ers = exp(r_val/s_val)
        if add_cut_exp!(m, m.r_exp[n], m.s_exp[n], m.t_exp[n], -ers, ers*(r_val/s_val-1))
            is_viol_prim = true
        end
    end

    for n in 1:m.num_sdp
        V_eig = eigfact!(Symmetric(getvalue(m.V_sdp[n])), -Inf, -m.tol_prim_infeas)
        if isempty(V_eig[:values])
            continue
        end
        is_infeas = true
        if !add_cuts
            continue
        end

        # Dual is sum_{j: lambda_j < 0} lamda_j V_j V_j'
        if add_cut_sdp!(m, m.V_sdp[n], V_eig[:vectors])
            is_viol_prim = true
        end
    end

    m.logs[:prim_cuts] += toq()
    if !is_infeas
        m.logs[:n_feas_mip] += 1
    end

    return (is_infeas, is_viol_prim)
end

# Remove near-zeros from data, return false if all values are near-zeros, warn if bad conditioning on vector
function clean_zeros!{N}(m, data::Array{Float64,N})
    min_nz = Inf
    max_nz = 0

    for j in 1:length(data)
        absj = abs(data[j])

        if absj < m.tol_zero
            data[j] = 0.
            continue
        end

        if absj < min_nz
            min_nz = absj
        end
        if absj > max_nz
            max_nz = absj
        end
    end

    if max_nz > m.tol_zero
        if max_nz/min_nz > 1e7
            warn("Numerically unstable dual vector encountered\n")
        end
        return true
    else
        return false
    end
end

# Add K* cuts for a SOC, where (t,v) is the vector of slacks, return true if a cut is violated by current solution
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
            @expression(m.model_mip, cut_expr, 2*(t_dual*t - vecdot(abs.(v_dual), a)))
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

# Add K* cuts for a ExpPrimal cone, where (r,s,t) is the vector of slacks, return true if a cut is violated by current solution
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

# Add K* cuts for a PSD cone, where V is the matrix of smat-space slacks, return true if a cut is violated by current solution
function add_cut_sdp!(m, V, eig_dual)
    # Remove near-zeros, return false if all near zero
    if !clean_zeros!(m, eig_dual)
        return false
    end

    (dim, num_eig) = size(eig_dual)
    is_viol = false

    if m.sdp_eig
        # Using PSD eigenvector cuts
        for j in 1:num_eig
            eig_j = eig_dual[:, j]

            if m.sdp_soc && !(m.mip_solver_drives && m.oa_started)
                # Using SDP SOC eig cuts
                # Over all diagonal entries i, exclude the largest one
                (_, i) = findmax(abs.(eig_j))

                # yz >= ||x||^2, y,z >= 0 <==> norm2(1/sqrt2*(y-z), 1/sqrt2*(-y+z), 2x) <= y + z, y,z >= 0
                y = V[i,i]
                z = sum(V[k,l]*eig_j[k]*eig_j[l] for k in 1:dim, l in 1:dim if (k!=i && l!=i))
                @constraint(m.model_mip, z >= 0)
                x = sum(V[k,i]*eig_j[k] for k in 1:dim if k!=i)
                @expression(m.model_mip, cut_expr, y + z - norm([sqrt2inv*(y - z), sqrt2inv*(-y + z), 2*x]))
                if add_cut!(m, cut_expr, m.logs[:SDP])
                    is_viol = true
                end
            else
                # Not using SDP SOC cuts
                @expression(m.model_mip, cut_expr, num_eig*vecdot(eig_j*eig_j', V))
                if add_cut!(m, cut_expr, m.logs[:SDP])
                    is_viol = true
                end
            end
        end
    else
        # Using full PSD cut
        if m.sdp_soc && !(m.mip_solver_drives && m.oa_started)
            # Using SDP SOC full cut
            # Over all diagonal entries i, exclude the largest one
            mat_dual = eig_dual * eig_dual'
            (_, i) = findmax(abs.(diag(mat_dual)))

            # yz >= ||x||^2, y,z >= 0 <==> norm2(1/sqrt2*(y-z), 1/sqrt2*(-y+z), 2x) <= y + z, y,z >= 0
            y = V[i,i]
            z = sum(V[k,l]*mat_dual[k,l] for k in 1:dim, l in 1:dim if (k!=i && l!=i))
            @constraint(m.model_mip, z >= 0)
            @expression(m.model_mip, x[j in 1:num_eig], sum((V[k,i]*eig_dual[k,j]) for k in 1:dim if k!=i))
            @expression(m.model_mip, cut_expr, y + z - norm([sqrt2inv*(y - z), sqrt2inv*(-y + z), (2*x)...]))
            if add_cut!(m, cut_expr, m.logs[:SDP])
                is_viol = true
            end
        else
            # Using PSD linear cut
            @expression(m.model_mip, cut_expr, vecdot(eig_dual*eig_dual', V))
            if add_cut!(m, cut_expr, m.logs[:SDP])
                is_viol = true
            end
        end
    end

    return is_viol
end

# Check and record violation and add cut, return true if violated
function add_cut!(m, cut_expr, cone_logs)
    if !m.oa_started
        @constraint(m.model_mip, cut_expr >= 0)
        cone_logs[:n_relax] += 1
        return false
    end

    if -getvalue(cut_expr) > m.tol_prim_infeas
        if m.mip_solver_drives
            @lazyconstraint(m.cb_lazy, cut_expr >= 0)
        else
            @constraint(m.model_mip, cut_expr >= 0)
        end
        cone_logs[:n_viol_total] += 1
        return true
    elseif !m.viol_cuts_only
        if m.mip_solver_drives
            @lazyconstraint(m.cb_lazy, cut_expr >= 0)
        else
            @constraint(m.model_mip, cut_expr >= 0)
        end
        cone_logs[:n_nonviol_total] += 1
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
    logs[:mip_solve] = 0.   # Solving the MIP model
    logs[:subp_solve] = 0. # Solving conic subproblem model
    logs[:relax_cuts] = 0.  # Deriving and adding conic relaxation cuts
    logs[:subp_cuts] = 0.   # Deriving and adding subproblem cuts
    logs[:prim_cuts] = 0.   # Deriving and adding primal cuts

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

    # Cuts counters
    for cone in (:SOC, :ExpPrimal, :SDP)
        logs[cone] = Dict{Symbol,Any}()
        logs[cone][:n_relax] = 0
        logs[cone][:n_viol_total] = 0
        logs[cone][:n_nonviol_total] = 0
    end

    m.logs = logs
end

# Print objective gap information for iterative
function print_gap(m)
    if m.log_level < 1
        return
    end

    if (m.logs[:n_iter] == 1) || (m.log_level > 2)
        @printf "\n%-4s | %-14s | %-14s | %-11s | %-11s\n" "Iter" "Best feasible" "Best bound" "Rel gap" "Time (s)"
    end
    if m.gap_rel_opt < 1000
        @printf "%4d | %+14.6e | %+14.6e | %11.3e | %11.3e\n" m.logs[:n_iter] m.best_obj m.mip_obj m.gap_rel_opt (time() - m.logs[:total])
    else
        @printf "%4d | %+14.6e | %+14.6e | %11s | %11.3e\n" m.logs[:n_iter] m.best_obj m.mip_obj (isnan(m.gap_rel_opt) ? "Inf" : ">1000") (time() - m.logs[:total])
    end
    flush(STDOUT)
end

# Print after finish
function print_finish(m::PajaritoConicModel)
    flush(STDOUT)

    if m.gap_rel_opt < -10*m.rel_gap
        if m.is_best_conic
            warn("Best feasible value is smaller than best bound: conic solver's solution may have significant infeasibilities (try tightening primal feasibility tolerance of conic solver)\n")
        else
            warn("Best feasible value is smaller than best bound: check solution feasibility and bounds returned by MIP solver (please submit an issue)\n")
        end
        # m.status = :Error
    end

    if (m.log_level > 0) && !in(m.status, [:Optimal, :Suboptimal, :UserLimit, :Unbounded, :Infeasible])
        m.log_level = 3
    end

    if m.log_level >= 1
        if m.mip_solver_drives
            @printf "\nMIP-solver-driven algorithm summary:\n"
        else
            @printf "\nIterative algorithm summary:\n"
        end
        @printf " - Status               = %14s\n" m.status
        @printf " - Best feasible        = %+14.6e\n" m.best_obj
        @printf " - Best bound           = %+14.6e\n" m.mip_obj
        if m.gap_rel_opt < -10*m.rel_gap
            @printf " - Relative opt. gap    =*%14.3e*\n" m.gap_rel_opt
        else
            @printf " - Relative opt. gap    = %14.3e\n" m.gap_rel_opt
        end
        @printf " - Total time (s)       = %14.2e\n" m.logs[:total]
    end

    if m.log_level >= 2
        @printf "Solution constructed by %s solver\n" (m.is_best_conic ? "conic" : "MIP")
    end

    if m.log_level >= 3
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
        @printf " -- Get primal cuts     = %10.2e\n" m.logs[:prim_cuts]

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
            @printf " --- From OA model      = %5d\n" m.logs[:n_feas_mip]
        else
            @printf " --- In lazy callback   = %5d\n" m.logs[:n_feas_mip]
            @printf " - Heuristic callbacks  = %5d\n" m.logs[:n_heur]
            @printf " -- Solutions passed    = %5d\n" m.logs[:n_add]
        end
    end

    if m.log_level >= 2
        @printf "\nOuter-approximation cuts added:"
        @printf "\n%-16s | %-9s | %-9s | %-9s\n" "Cone" "Relax." "Violated" "Nonviol."
        for (cone, name) in zip((:SOC, :ExpPrimal, :SDP), ("Second order", "Primal expon.", "Pos. semidef."))
            log = m.logs[cone]
            if (log[:n_relax] + log[:n_viol_total] + log[:n_nonviol_total]) > 0
                @printf "%16s | %9d | %9d | %9d\n" name log[:n_relax] log[:n_viol_total] log[:n_nonviol_total]
            end
        end

        if isfinite(m.best_obj) && !any(isnan, m.final_soln)
            var_inf = calc_infeas(m.cone_var_orig, m.final_soln)
            con_inf = calc_infeas(m.cone_con_orig, m.b_orig-m.A_orig*m.final_soln)

            @printf "\nDistance to feasibility (negative indicates strict feasibility):"
            @printf "\n%-16s | %-9s | %-10s\n" "Cone" "Variable" "Constraint"
            for (v, c, name) in zip(var_inf, con_inf, ("Linear", "Second order", "Rotated S.O.", "Primal expon.", "Pos. semidef."))
                if isfinite(v) && isfinite(c)
                    @printf "%16s | %9.2e | %9.2e\n" name -v -c
                elseif isfinite(v)
                    @printf "%16s | %9.2e | %9s\n" name -v "NA"
                elseif isfinite(c)
                    @printf "%16s | %9s | %9.2e\n" name "NA" -c
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

    flush(STDOUT)
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
            viol_soc = max(viol_soc, vecnorm(vals[idx[j]] for j in 2:length(idx)) - vals[idx[1]])
        elseif cone == :SOCRotated
            # (y,z,x) in RSOC <=> (y+z,-y+z,sqrt2*x) in SOC, y >= 0, z >= 0
            viol_rot = max(viol_rot, sqrt((vals[idx[1]] - vals[idx[2]])^2 + 2. * sumabs2(vals[idx[j]] for j in 3:length(idx))) - (vals[idx[1]] + vals[idx[2]]))
        elseif cone == :ExpPrimal
            viol_exp = max(viol_exp, vals[idx[2]]*exp(vals[idx[1]]/vals[idx[2]]) - vals[idx[3]])
        elseif cone == :SDP
            dim = round(Int, sqrt(1/4+2*length(idx))-1/2) # smat space dimension
            vals_smat = Symmetric(zeros(dim, dim))
            make_smat!(vals_smat, vals[idx])
            viol_sdp = max(viol_sdp, -eigmin(vals_smat))
        end
    end

    return (viol_lin, viol_soc, viol_rot, viol_exp, viol_sdp)
end
