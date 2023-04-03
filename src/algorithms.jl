# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# outer approximation algorithms

function optimize(opt::Optimizer)
    start_optimize(opt)

    # setup models
    setup_finish = setup_models(opt)
    if setup_finish
        finish_optimize(opt)
        return
    end

    if opt.solve_relaxation
        # solve continuous relaxation
        relax_finish = solve_relaxation(opt)
        if relax_finish
            finish_optimize(opt)
            return
        end

        # add continuous relaxation cuts
        add_relax_cuts(opt)
    end

    # add variable bounds and (optionally) initial fixed cuts
    add_init_cuts(opt)

    # ensure continuous relaxation of OA model is bounded
    num_relax_iters = 0
    while num_relax_iters < 10
        finish_bound = run_oa_relax_bound(opt)
        finish_bound && break
        num_relax_iters += 1
    end
    if opt.verbose
        println("separated $num_relax_iters rays before imposing integrality")
    end

    # add discrete constraints to OA model
    add_discrete_constraints(opt)

    # run main OA algorithm
    if opt.use_iterative_method
        print_header(opt)
        while true
            opt.num_iters += 1
            finish_iter = run_iterative_method(opt)
            finish_iter && break
        end
    else
        run_one_tree_method(opt)
    end

    finish_optimize(opt)
    return
end

# one iteration of the iterative method
function run_iterative_method(opt::Optimizer)
    # solve OA model
    time_finish = check_set_time_limit(opt, opt.oa_model)
    time_finish && return true
    JuMP.optimize!(opt.oa_model)

    oa_status = JuMP.termination_status(opt.oa_model)
    if oa_status == MOI.INFEASIBLE
        if opt.verbose
            println("infeasibility detected while iterating; terminating")
        end
        opt.status = oa_status
        return true
    elseif oa_status == MOI.TIME_LIMIT
        if opt.verbose
            println("OA solver timed out")
        end
        opt.status = oa_status
        return true
    elseif oa_status != MOI.OPTIMAL
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
        return true
    end

    # update objective bound
    obj_bound = get_objective_bound(opt.oa_model)
    if !isfinite(obj_bound) && oa_status == MOI.OPTIMAL
        obj_bound = JuMP.objective_value(opt.oa_model)
    end
    opt.obj_bound = obj_bound

    subp_cuts_added = false
    if opt.solve_subproblems
        # solve conic subproblem with fixed integer solution and update incumbent
        # check if integer solution is repeated
        int_sol = get_integral_solution(opt)
        hash_int_sol = hash(int_sol)
        if haskey(opt.int_sols_cuts, hash_int_sol)
            @warn("integral solution repeated")
        else
            # new integral solution: solve subproblem and add cuts
            opt.int_sols_cuts[hash_int_sol] = JuMP.AffExpr[]
            subp_failed = solve_subproblem(int_sol, opt)
            if !subp_failed
                subp_cuts_added = add_subp_cuts(opt, false, nothing)
            end
        end
    end

    if !subp_cuts_added
        # add separation cuts and update incumbent from OA solution if no cuts are added
        # TODO should check feas whether or not cuts are added, and accept incumbent if feas
        cuts_added = add_sep_cuts(opt)
        if !cuts_added
            # no separation cuts, so try updating incumbent from OA solver
            update_incumbent_from_OA(opt)
        end
    end

    # print and check convergence
    obj_rel_gap = get_obj_rel_gap(opt)
    if opt.verbose
        Printf.@printf(
            "%5d %8d %12.4e %12.4e %12.4e\n",
            opt.num_iters,
            opt.num_cuts,
            opt.obj_value,
            opt.obj_bound,
            obj_rel_gap,
        )
    end
    if !isnan(obj_rel_gap) && obj_rel_gap < opt.tol_rel_gap
        if opt.verbose
            println("objective relative gap $obj_rel_gap reached; terminating")
        end
        opt.status = MOI.OPTIMAL
        return true
    end
    obj_abs_gap = get_obj_abs_gap(opt)
    if !isnan(obj_abs_gap) && obj_abs_gap < opt.tol_abs_gap
        if opt.verbose
            println("objective absolute gap $obj_abs_gap reached; terminating")
        end
        opt.status = MOI.OPTIMAL
        return true
    end

    # check iteration limit
    if opt.num_iters == opt.iteration_limit
        if opt.verbose
            println("iteration limit ($(opt.num_iters)) reached; terminating")
        end
        opt.status = MOI.ITERATION_LIMIT
        return true
    end

    # set warm start from incumbent
    if opt.use_oa_starts && isfinite(opt.obj_value)
        oa_start = get_oa_start(opt, opt.incumbent)
        JuMP.set_start_value.(opt.oa_vars, oa_start)
    end
    return false
end

# one tree method using callbacks
function run_one_tree_method(opt::Optimizer)
    if opt.verbose
        println("starting one tree method")
    end
    oa_model = opt.oa_model
    if iszero(opt.num_int_vars) && isempty(opt.SOS12_cons)
        println(
            "model has no discrete variables, so adding a dummy integer variable",
        )
        dummy = JuMP.@variable(oa_model, integer = true)
    else
        dummy = nothing
    end

    function lazy_cb(cb)
        opt.num_lazy_cbs += 1
        opt.lazy_cb = cb
        int = (
            JuMP.callback_node_status(cb, oa_model) ==
            MOI.CALLBACK_NODE_STATUS_INTEGER
        )

        subp_cuts_added = false
        if opt.solve_subproblems
            # only solve subproblem at an integer solution
            int || return

            # check if integer solution is repeated and cache the cuts
            int_sol = get_integral_solution(opt)
            hash_int_sol = hash(int_sol)
            if haskey(opt.int_sols_cuts, hash_int_sol)
                # integral solution repeated: add cached cuts
                cuts = opt.int_sols_cuts[hash_int_sol]
                subp_cuts_added = add_cuts(cuts, opt, true)
                if !subp_cuts_added && opt.verbose
                    println("cached subproblem cuts could not be added")
                end
            else
                # new integral solution: solve subproblem, cache cuts, and add cuts
                cuts_cache = opt.int_sols_cuts[hash_int_sol] = JuMP.AffExpr[]
                subp_failed = solve_subproblem(int_sol, opt)
                if !subp_failed
                    subp_cuts_added = add_subp_cuts(opt, true, cuts_cache)
                end
            end
        end

        if !subp_cuts_added
            cuts_added = add_sep_cuts(opt)
            if !cuts_added && int
                # integer solution and no separation cuts added, so try updating incumbent
                update_incumbent_from_OA(opt)
            end
        end
        return
    end
    MOI.set(oa_model, MOI.LazyConstraintCallback(), lazy_cb)

    if opt.solve_subproblems
        function heuristic_cb(cb)
            opt.num_heuristic_cbs += 1
            opt.new_incumbent || return

            oa_start = get_oa_start(opt, opt.incumbent)
            status = MOI.submit(
                oa_model,
                MOI.HeuristicSolution(cb),
                opt.oa_vars,
                oa_start,
            )
            if opt.verbose
                println("heuristic cb status was: ", status)
            end
            opt.new_incumbent = false
            return
        end
        MOI.set(oa_model, MOI.HeuristicCallback(), heuristic_cb)
    end

    time_finish = check_set_time_limit(opt, oa_model)
    time_finish && return true
    JuMP.optimize!(oa_model)
    opt.lazy_cb = nothing

    oa_status = JuMP.termination_status(oa_model)
    if oa_status == MOI.OPTIMAL
        opt.status = oa_status
        opt.obj_bound = get_objective_bound(oa_model)

        if !isfinite(opt.obj_value)
            # use OA solver solution TODO should check feasibility
            @warn("taking OA solver solution, which may be infeasible")
            update_incumbent_from_OA(opt)
        end
    elseif oa_status == MOI.INFEASIBLE
        opt.status = oa_status
    elseif oa_status == MOI.TIME_LIMIT
        if opt.verbose
            println("OA solver timed out")
        end
        opt.status = oa_status
    else
        @warn("OA solver status $oa_status is not handled")
        opt.status = MOI.OTHER_ERROR
    end

    if !isnothing(dummy)
        # delete dummy integer constraint
        JuMP.delete(oa_model, dummy)
    end
    return
end

function solve_relaxation(opt::Optimizer)
    if opt.verbose
        println("solving continuous relaxation")
    end
    relax_model = opt.relax_model
    time_finish = check_set_time_limit(opt, relax_model)
    time_finish && return true
    JuMP.optimize!(relax_model)

    relax_status = JuMP.termination_status(relax_model)
    if opt.verbose
        println("continuous relaxation status is $relax_status")
    end
    if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        opt.obj_bound = JuMP.dual_objective_value(relax_model)
    elseif relax_status in (MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE)
        if opt.verbose
            println("problem could be unbounded; Pajarito may fail to converge")
        end
    elseif relax_status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
        if opt.verbose
            println(
                "infeasibility detected from continuous relaxation; terminating",
            )
        end
        opt.status = MOI.INFEASIBLE
        return true
    elseif relax_status == MOI.TIME_LIMIT
        if opt.verbose
            println("continuous relaxation solver timed out")
        end
        opt.status = relax_status
        return true
    else
        @warn("continuous relaxation status $relax_status is not handled")
        return false
    end

    finish = false

    # check whether problem is continuous
    if iszero(opt.num_int_vars) && isempty(opt.SOS12_cons)
        if opt.verbose
            println("problem is continuous; terminating")
        end
        finish = true
    end

    if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        # check whether solution is feasible for integrality and SOS1/2 constraints
        if is_discrete_feas(opt, opt.relax_x)
            if opt.verbose
                println(
                    "relaxation solution satisfies discrete constraints; terminating",
                )
            end
            finish = true
        end
    end

    if finish
        if relax_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
            opt.status = MOI.OPTIMAL
            opt.obj_value = JuMP.objective_value(relax_model)
            opt.incumbent = JuMP.value.(opt.relax_x)
        elseif relax_status in (MOI.DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE)
            opt.status = MOI.DUAL_INFEASIBLE
        else
            error("status $(opt.status) not handled")
        end
        return true
    end
    return false
end

# solve subproblem with new integer variable sub-solution
function solve_subproblem(int_sol::Vector{Int}, opt::Optimizer)
    # update conic data for int_sol
    modify_subproblem(int_sol, opt)

    # solve
    subp_model = opt.subp_model
    time_finish = check_set_time_limit(opt, subp_model)
    time_finish && return true
    JuMP.optimize!(subp_model)

    subp_status = JuMP.termination_status(subp_model)
    if opt.verbose
        println("continuous subproblem status is $subp_status")
    end
    if subp_status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        obj_val =
            JuMP.objective_value(subp_model) +
            LinearAlgebra.dot(opt.c_int, int_sol)
        if obj_val < opt.obj_value
            # update incumbent and objective value
            subp_sol = JuMP.value.(opt.subp_x)
            opt.incumbent = vcat(int_sol, subp_sol)
            opt.obj_value = obj_val
            # println("new incumbent")
            opt.new_incumbent = true
        end
        return false
    elseif subp_status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
        # NOTE: duals are rescaled before adding subproblem cuts
        return false
    elseif subp_status == MOI.TIME_LIMIT
        if opt.verbose
            println("continuous subproblem solver timed out")
        end
        opt.status = subp_status
        return true
    else
        @warn("continuous subproblem status $subp_status is not handled")
        return false
    end
end

# add separation cuts on primal rays of the continuous relaxation of the OA model
function run_oa_relax_bound(opt::Optimizer)
    # solve OA model
    time_finish = check_set_time_limit(opt, opt.oa_model)
    time_finish && return true
    JuMP.optimize!(opt.oa_model)

    pr_status = JuMP.primal_status(opt.oa_model)
    if pr_status in
       (MOI.INFEASIBILITY_CERTIFICATE, MOI.NEARLY_INFEASIBILITY_CERTIFICATE)
        # add separation cuts for primal ray
        cuts_added = add_sep_cuts(opt)
        cuts_added && return false
        @warn("continuous relaxation of OA model could not be bounded")
    end
    return true
end

# update incumbent from OA solver
function update_incumbent_from_OA(opt::Optimizer)
    sol = get_value(opt.oa_x, opt.lazy_cb)
    obj_val = LinearAlgebra.dot(opt.c, sol)
    if obj_val < opt.obj_value
        # update incumbent and objective value
        opt.incumbent = sol
        opt.obj_value = obj_val
        println("new incumbent")
        opt.new_incumbent = true
    end
    return
end

function get_integral_solution(opt::Optimizer)
    x_int = opt.oa_x[1:(opt.num_int_vars)]
    int_sol = get_value(x_int, opt.lazy_cb)

    # check solution is integral
    round_int_sol = round.(Int, int_sol)
    # TODO different tol option?
    if !isapprox(round_int_sol, int_sol, atol = opt.tol_feas)
        @warn("integer variable solution is not integral to tolerance tol_feas")
    end
    return round_int_sol
end

function is_discrete_feas(opt::Optimizer, vars::Vector{JuMP.VariableRef})
    # check whether solution is integral
    int_sol = JuMP.value.(vars[1:(opt.num_int_vars)])
    round_int_sol = round.(Int, int_sol)
    if !isapprox(round_int_sol, int_sol, atol = opt.tol_feas)
        return false
    end

    # check whether SOS1/2 constraints are satisfied
    for (idxs, si) in opt.SOS12_cons
        num_nz = count(i -> abs(JuMP.value(vars[i])) > opt.tol_feas, idxs)
        if num_nz > (si isa MOI.SOS1 ? 1 : 2)
            return false
        end
    end
    return true
end

# initialize and print
function start_optimize(opt::Optimizer)
    get_conic_opt(opt)
    get_oa_opt(opt)
    empty_optimize(opt)
    opt.solve_time = time()
    opt.obj_value = Inf
    opt.obj_bound = -Inf
    return
end

# finalize and print
function finish_optimize(opt::Optimizer)
    opt.solve_time = time() - opt.solve_time

    oa_status = MOI.get(opt.oa_opt, MOI.TerminationStatus())
    if opt.verbose && oa_status != MOI.OPTIMIZE_NOT_CALLED
        println(
            "OA solver finished with status $oa_status, after $(opt.solve_time) " *
            "seconds and $(opt.num_cuts) cuts",
        )
        if opt.use_iterative_method
            println("iterative method used $(opt.num_iters) iterations")
        else
            println(
                "one tree method used $(opt.num_lazy_cbs) lazy " *
                "callbacks and $(opt.num_heuristic_cbs) heuristic callbacks",
            )
        end
    end
    opt.verbose && println()
    return
end

# compute objective absolute gap
function get_obj_abs_gap(opt::Optimizer)
    if opt.obj_sense == MOI.FEASIBILITY_SENSE
        return NaN
    end
    return opt.obj_value - opt.obj_bound
end

# compute objective relative gap
function get_obj_rel_gap(opt::Optimizer)
    return get_obj_abs_gap(opt) / (1e-5 + abs(opt.obj_value))
end

# print iteration statistics column names
function print_header(opt::Optimizer)
    opt.verbose || return
    Printf.@printf(
        "%5s %8s %12s %12s %12s\n",
        "iter",
        "cuts",
        "obj",
        "bound",
        "gap"
    )
    return
end
