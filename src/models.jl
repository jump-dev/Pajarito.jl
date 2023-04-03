# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# conic and OA models for OA algorithms

# setup conic and OA models
function setup_models(opt::Optimizer)
    @assert !opt.solve_subproblems || isempty(opt.SOS12_cons)

    # mixed-integer OA model, add discrete constraints later
    oa_model = opt.oa_model = JuMP.Model(() -> opt.oa_opt)
    oa_x = opt.oa_x = JuMP.@variable(oa_model, [1:length(opt.c)])
    JuMP.@objective(oa_model, Min, LinearAlgebra.dot(opt.c, oa_x))
    JuMP.@constraint(oa_model, opt.A * oa_x .== opt.b)
    oa_aff = JuMP.@expression(oa_model, opt.h - opt.G * oa_x)

    if opt.solve_relaxation
        # continuous relaxation model
        relax_model = opt.relax_model = JuMP.Model(() -> opt.conic_opt)
        relax_x = opt.relax_x = JuMP.@variable(relax_model, [1:length(opt.c)])
        JuMP.@objective(relax_model, Min, LinearAlgebra.dot(opt.c, relax_x))
        JuMP.@constraint(
            relax_model,
            opt.b - opt.A * relax_x in MOI.Zeros(length(opt.b))
        )
        relax_aff = JuMP.@expression(relax_model, opt.h - opt.G * relax_x)
    end

    if opt.solve_subproblems
        # differentiate integer and continuous variables
        num_cont_vars = length(opt.c) - opt.num_int_vars
        int_range = 1:(opt.num_int_vars)
        cont_range = (opt.num_int_vars+1):length(opt.c)
        opt.c_int = opt.c[int_range]
        opt.G_int = opt.G[:, int_range]
        c_cont = opt.c[cont_range]
        G_cont = opt.G[:, cont_range]
        A_cont = opt.A[:, cont_range]

        # remove zero rows in A_cont for subproblem
        keep_rows = (vec(maximum(abs, A_cont, dims = 2)) .>= 1e-10)
        A_cont = A_cont[keep_rows, :]
        opt.b_cont = opt.b[keep_rows]
        opt.A_int = opt.A[keep_rows, int_range]

        # continuous subproblem model
        subp_model = opt.subp_model = JuMP.Model(() -> opt.conic_opt)
        subp_x = opt.subp_x = JuMP.@variable(subp_model, [1:num_cont_vars])
        JuMP.@objective(subp_model, Min, LinearAlgebra.dot(c_cont, subp_x))
        K0 = MOI.Zeros(length(opt.b_cont))
        opt.subp_eq = JuMP.@constraint(subp_model, -A_cont * subp_x in K0)
        subp_aff = JuMP.@expression(subp_model, -G_cont * subp_x)
    end

    oa_vars = opt.oa_vars = copy(oa_x)
    opt.subp_cones = JuMP.ConstraintRef[]
    opt.subp_cone_idxs = UnitRange{Int}[]
    opt.relax_oa_cones = JuMP.ConstraintRef[]
    opt.subp_oa_cones = JuMP.ConstraintRef[]
    opt.cone_caches = Cache[]
    opt.oa_cone_idxs = UnitRange{Int}[]
    opt.oa_slack_idxs = UnitRange{Int}[]
    opt.unique_cones = Dict{UInt,Any}()

    for (cone, idxs) in zip(opt.cones, opt.cone_idxs)
        oa_supports = MOI.supports_constraint(
            opt.oa_opt,
            MOI.VectorAffineFunction{Float64},
            typeof(cone),
        )

        if opt.solve_relaxation
            relax_cone_i =
                JuMP.@constraint(relax_model, relax_aff[idxs] in cone)
            if !oa_supports
                push!(opt.relax_oa_cones, relax_cone_i)
            end
        end

        if opt.solve_subproblems
            subp_aff_i = subp_aff[idxs]
            if !oa_supports || !iszero(subp_aff_i)
                # conic constraint must be added to subproblem
                subp_cone_i = JuMP.@constraint(subp_model, subp_aff_i in cone)
                push!(opt.subp_cones, subp_cone_i)
                push!(opt.subp_cone_idxs, idxs)
            end
            if !oa_supports
                push!(opt.subp_oa_cones, subp_cone_i)
            end
        end

        oa_aff_i = oa_aff[idxs]
        if oa_supports
            JuMP.@constraint(oa_model, oa_aff_i in cone)
        else
            # add slack variables where useful and modify oa_aff_i
            (slacks, slack_idxs) = create_slacks(oa_model, oa_aff_i)
            append!(oa_vars, slacks)
            push!(opt.oa_slack_idxs, idxs[slack_idxs])

            # set up cone cache and extended formulation
            cache = Cones.create_cache(oa_aff_i, cone, opt)
            ext_i = Cones.setup_auxiliary(cache, opt)
            append!(oa_vars, ext_i)
            push!(opt.cone_caches, cache)
            push!(opt.oa_cone_idxs, idxs)
        end
    end
    @assert JuMP.num_variables(oa_model) == length(oa_vars)

    opt.use_oa_starts = MOI.supports(
        JuMP.backend(oa_model),
        MOI.VariablePrimalStart(),
        MOI.VariableIndex,
    )
    if opt.use_oa_starts && !isempty(opt.warm_start)
        if any(isnan, opt.warm_start)
            @warn("warm start is only partial so will be ignored")
        else
            oa_start = get_oa_start(opt, opt.warm_start)
            JuMP.set_start_value.(oa_vars, oa_start)
        end
    end

    isempty(opt.cone_caches) || return false
    # no conic constraints need outer approximation, so just solve the OA model and finish
    if opt.verbose
        println("no conic constraints need outer approximation")
    end

    # add integrality constraints to OA model and solve
    add_discrete_constraints(opt)
    time_finish = check_set_time_limit(opt, oa_model)
    time_finish && return true
    JuMP.optimize!(oa_model)

    opt.status = JuMP.termination_status(oa_model)
    if opt.status == MOI.OPTIMAL
        opt.obj_value = JuMP.objective_value(oa_model)
        opt.obj_bound = get_objective_bound(oa_model)
        opt.incumbent = JuMP.value.(oa_x)
    end
    return true
end

# to balance variable dimension and sparsity of the constraint matrix with K* cuts, only add
# slacks if number of variables involved in this constraint exceeds the constraint dimension
function create_slacks(model::JuMP.Model, expr_vec::Vector{JuMP.AffineExpr})
    slacks = JuMP.VariableRef[]
    slack_idxs = Int[]
    vars = Set(k.index.value for f in expr_vec for k in keys(f.terms))
    if length(vars) > length(expr_vec)
        # number of variables in expr_vec exceeds dimension of expr_vec
        for (j, expr_j) in enumerate(expr_vec)
            terms = JuMP.linear_terms(expr_j)
            length(terms) <= 1 && continue

            # affine expression has more than one variable, so add a slack variable
            s_j = JuMP.@variable(model)
            JuMP.@constraint(model, s_j .== expr_j)
            expr_vec[j] = s_j
            push!(slacks, s_j)
            push!(slack_idxs, j)
        end
    end
    return (slacks, slack_idxs)
end

function get_oa_start(opt::Optimizer, x_start::Vector{Float64})
    n = length(opt.incumbent)
    @assert length(x_start) == n
    oa_start = fill(NaN, length(opt.oa_vars))
    oa_start[1:n] .= x_start

    s_start = opt.h - opt.G * x_start
    for (i, cache) in enumerate(opt.cone_caches)
        slack_idxs = opt.oa_slack_idxs[i]
        if !isempty(slack_idxs)
            # set slack variables start
            slack_start = s_start[slack_idxs]
            dim = length(slack_start)
            oa_start[n.+(1:dim)] .= slack_start
            n += dim
        end

        ext_dim = Cones.num_ext_variables(cache)
        if !iszero(ext_dim)
            # set auxiliary variables start
            s_start_i = s_start[opt.oa_cone_idxs[i]]
            ext_start = Cones.extend_start(cache, s_start_i, opt)
            @assert ext_dim == length(ext_start)
            oa_start[n.+(1:ext_dim)] .= ext_start
            n += ext_dim
        end
    end
    @assert n == length(oa_start)

    @assert !any(isnan, oa_start)
    return oa_start
end

function modify_subproblem(int_sol::Vector{Int}, opt::Optimizer)
    # TODO maybe also modify the objective constant using dot(opt.c_int, int_sol), could be nonzero
    moi_model = JuMP.backend(opt.subp_model)

    new_b = opt.b_cont - opt.A_int * int_sol
    MOI.modify(
        moi_model,
        JuMP.index(opt.subp_eq),
        MOI.VectorConstantChange(new_b),
    )

    new_h = opt.h - opt.G_int * int_sol
    for (cr, idxs) in zip(opt.subp_cones, opt.subp_cone_idxs)
        MOI.modify(
            moi_model,
            JuMP.index(cr),
            MOI.VectorConstantChange(new_h[idxs]),
        )
    end
    return
end

function add_discrete_constraints(opt::Optimizer)
    JuMP.set_integer.(opt.oa_x[1:(opt.num_int_vars)])
    for (idxs, si) in opt.SOS12_cons
        JuMP.@constraint(opt.oa_model, opt.oa_x[idxs] in si)
    end
    return
end
