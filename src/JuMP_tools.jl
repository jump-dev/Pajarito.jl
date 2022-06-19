# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# tools for JuMP functions

function add_cuts(cuts::Vector{AE}, opt::Optimizer, viol_only::Bool)
    if viol_only
        # filter out unviolated cuts
        cuts = filter(cut -> get_value(cut, opt.lazy_cb) < -opt.tol_feas, cuts)
    end

    _add_cuts(cuts, opt.oa_model, opt.lazy_cb)
    opt.num_cuts += length(cuts)
    return !isempty(cuts)
end

function _add_cuts(cuts::Vector{AE}, model::JuMP.Model, ::Nothing)
    JuMP.@constraint(model, cuts .>= 0)
    return
end

function _add_cuts(cuts::Vector{AE}, model::JuMP.Model, cb)
    cons = JuMP.@build_constraint(cuts .>= 0)
    MOI.submit.(model, MOI.LazyConstraint(cb), cons)
    return
end

get_value(expr::Union{VR,AE}, ::Nothing) = JuMP.value(expr)

get_value(expr::Union{VR,AE}, cb) = JuMP.callback_value(cb, expr)

get_value(exprs::Vector{<:Union{VR,AE}}, cb) = [get_value(e, cb) for e in exprs]

function check_set_time_limit(opt::Optimizer, model::JuMP.Model)
    time_left = opt.time_limit - time() + opt.solve_time
    if time_left < 1e-3
        if opt.verbose
            println("time limit ($(opt.time_limit)) reached; terminating")
        end
        opt.status = MOI.TIME_LIMIT
        return true
    end
    if MOI.supports(JuMP.backend(model), MOI.TimeLimitSec())
        JuMP.set_time_limit_sec(model, time_left)
    end
    return false
end

# try to get an objective bound or dual objective value, else use the primal objective value
function get_objective_bound(model::JuMP.Model)
    try
        return JuMP.objective_bound(model)
    catch
    end
    try
        return JuMP.dual_objective_value(model)
    catch
    end
    return JuMP.objective_value(model)
end

# get number of constraints of all types
function get_num_constraints(model::JuMP.Model)
    num_cons = 0
    for (F, S) in JuMP.list_of_constraint_types(model)
        num_cons += JuMP.num_constraints(model, F, S)
    end
    return num_cons
end
