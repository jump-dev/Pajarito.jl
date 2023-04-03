# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# optimizer struct

mutable struct Optimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool
    tol_feas::Float64
    tol_rel_gap::Float64
    tol_abs_gap::Float64
    time_limit::Float64
    iteration_limit::Int
    use_iterative_method::Union{Nothing,Bool}
    use_extended_form::Bool
    solve_relaxation::Bool
    solve_subproblems::Bool
    use_init_fixed_oa::Bool
    oa_solver::Union{Nothing,MOI.OptimizerWithAttributes}
    conic_solver::Union{Nothing,MOI.OptimizerWithAttributes}

    # optimizers
    oa_opt::Union{Nothing,MOI.AbstractOptimizer}
    conic_opt::Union{Nothing,MOI.AbstractOptimizer}

    # problem data
    obj_sense::MOI.OptimizationSense
    obj_offset::Float64
    c::Vector{Float64}
    A::SparseArrays.SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    G::SparseArrays.SparseMatrixCSC{Float64,Int}
    h::Vector{Float64}
    cones::Vector{MOI.AbstractVectorSet}
    cone_idxs::Vector{UnitRange{Int}}
    num_int_vars::Int
    SOS12_cons::Vector{
        Pair{Vector{Int},<:Union{MOI.SOS1{Float64},MOI.SOS2{Float64}}},
    }

    # models, variables, etc
    oa_model::JuMP.Model
    relax_model::JuMP.Model
    subp_model::JuMP.Model
    oa_x::Vector{JuMP.VariableRef}
    relax_x::Vector{JuMP.VariableRef}
    subp_x::Vector{JuMP.VariableRef}
    subp_eq::JuMP.ConstraintRef
    subp_cones::Vector{JuMP.ConstraintRef}
    subp_cone_idxs::Vector{UnitRange{Int}}
    c_int::Vector{Float64}
    A_int::SparseArrays.SparseMatrixCSC{Float64,Int}
    G_int::SparseArrays.SparseMatrixCSC{Float64,Int}
    b_cont::Vector{Float64}
    oa_vars::Vector{JuMP.VariableRef}
    relax_oa_cones::Vector{JuMP.ConstraintRef}
    subp_oa_cones::Vector{JuMP.ConstraintRef}
    cone_caches::Vector{Cache}
    oa_cone_idxs::Vector{UnitRange{Int}}
    oa_slack_idxs::Vector{Vector{Int}}

    # used/modified during optimize
    lazy_cb::Any
    new_incumbent::Bool
    int_sols_cuts::Dict{UInt,Vector{JuMP.AffineExpr}}
    use_oa_starts::Bool

    # used by MOI wrapper
    incumbent::Vector{Float64}
    warm_start::Vector{Float64}
    status::MOI.TerminationStatusCode
    solve_time::Float64
    obj_value::Float64
    obj_bound::Float64
    num_cuts::Int
    num_iters::Int
    num_lazy_cbs::Int
    num_heuristic_cbs::Int

    # useful for PajaritoExtras
    sep_solver::Union{Nothing,MOI.OptimizerWithAttributes}
    unique_cones::Dict{UInt,Any}

    function Optimizer(
        verbose::Bool = true,
        tol_feas::Float64 = 1e-7,
        tol_rel_gap::Float64 = 1e-5,
        tol_abs_gap::Float64 = 1e-4,
        time_limit::Float64 = 1e6,
        iteration_limit::Int = 1000,
        use_iterative_method::Union{Nothing,Bool} = nothing,
        use_extended_form::Bool = true,
        solve_relaxation::Bool = true,
        solve_subproblems::Bool = true,
        use_init_fixed_oa::Bool = true,
        oa_solver::Union{Nothing,MOI.OptimizerWithAttributes} = nothing,
        conic_solver::Union{Nothing,MOI.OptimizerWithAttributes} = nothing,
        sep_solver::Union{Nothing,MOI.OptimizerWithAttributes} = nothing,
    )
        opt = new()
        opt.verbose = verbose
        opt.tol_feas = tol_feas
        opt.tol_rel_gap = tol_rel_gap
        opt.tol_abs_gap = tol_abs_gap
        opt.time_limit = time_limit
        opt.iteration_limit = iteration_limit
        opt.use_iterative_method = use_iterative_method
        opt.use_extended_form = use_extended_form
        opt.solve_relaxation = solve_relaxation
        opt.solve_subproblems = solve_subproblems
        opt.use_init_fixed_oa = use_init_fixed_oa
        opt.oa_solver = oa_solver
        opt.conic_solver = conic_solver
        opt.oa_opt = nothing
        opt.conic_opt = nothing
        opt.sep_solver = sep_solver
        return empty_optimize(opt)
    end
end

function empty_optimize(opt::Optimizer)
    opt.status = MOI.OPTIMIZE_NOT_CALLED
    opt.solve_time = NaN
    opt.obj_value = NaN
    opt.obj_bound = NaN
    opt.num_cuts = 0
    opt.num_iters = 0
    opt.num_lazy_cbs = 0
    opt.num_heuristic_cbs = 0
    opt.lazy_cb = nothing
    opt.new_incumbent = false
    opt.int_sols_cuts = Dict{UInt,Vector{JuMP.AffineExpr}}()

    if !isnothing(opt.oa_opt)
        MOI.empty!(opt.oa_opt)
    end
    if !isnothing(opt.conic_opt)
        MOI.empty!(opt.conic_opt)
    end
    return opt
end

# check/instantiate OA solver and set default for use_iterative_method
function get_oa_opt(opt::Optimizer)
    if isnothing(opt.oa_opt)
        if isnothing(opt.oa_solver)
            error("No outer approximation solver specified (set `oa_solver`)")
        end
        opt.oa_opt = MOI.instantiate(opt.oa_solver, with_bridge_type = Float64)

        # check whether lazy constraints are supported
        supports_lazy = MOI.supports(opt.oa_opt, MOI.LazyConstraintCallback())
        if isnothing(opt.use_iterative_method)
            # default to one tree method if possible
            opt.use_iterative_method = !supports_lazy
        elseif !opt.use_iterative_method && !supports_lazy
            error(
                "Outer approximation solver (`oa_solver`) does not support " *
                "lazy constraint callbacks (`use_iterative_method` must be `true`)",
            )
        end
    end
    return opt.oa_opt
end

# check/instantiate conic solver
function get_conic_opt(opt::Optimizer)
    if isnothing(opt.conic_opt)
        if isnothing(opt.conic_solver)
            error("No primal-dual conic solver specified (set `conic_solver`)")
        end
        opt.conic_opt =
            MOI.instantiate(opt.conic_solver, with_bridge_type = Float64)
    end
    return opt.conic_opt
end
