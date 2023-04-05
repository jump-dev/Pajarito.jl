# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# MathOptInterface wrapper of Pajarito solver

_is_empty(::Nothing) = true
_is_empty(opt::MOI.AbstractOptimizer) = MOI.is_empty(opt)

function MOI.is_empty(opt::Optimizer)
    return (_is_empty(opt.oa_opt) && _is_empty(opt.conic_opt))
end

MOI.empty!(opt::Optimizer) = empty_optimize(opt)

MOI.get(::Optimizer, ::MOI.SolverName) = "Pajarito"

MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.verbose = !value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Union{Real,Nothing})
    opt.time_limit = something(value, Inf)
    return
end

function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    if isfinite(opt.time_limit)
        return opt.time_limit
    end
    return
end

function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    if isnan(opt.solve_time)
        error("solve has not been called")
    end
    return opt.solve_time
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.status)

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return setproperty!(opt, Symbol(param.name), value)
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(opt, Symbol(param.name))
end

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction{<:Union{VI,SAF}}},
)
    return true
end

function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{VI},
    S::Type{MOI.Integer},
)
    return MOI.supports_constraint(get_oa_opt(opt), F, S)
end

# if not solving conic subproblems, allow SOS1/2 constraints if supported by OA solver
function MOI.supports_constraint(opt::Optimizer, F::Type{VV}, S::Type{<:SOS12})
    return !opt.solve_subproblems &&
           MOI.supports_constraint(get_oa_opt(opt), F, S)
end

# if using conic solver, cone must be supported by both Pajarito and the conic solver
function MOI.supports_constraint(
    opt::Optimizer,
    F::Type{<:Union{VV,VAF}},
    S::Type{<:Union{MOI.Zeros,MOI.Nonnegatives,Cones.OACone}},
)
    if !(opt.solve_relaxation || opt.solve_subproblems)
        return true
    end
    return MOI.supports_constraint(get_conic_opt(opt), F, S)
end

MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{VI}) = true

MOI.optimize!(opt::Optimizer) = optimize(opt)

MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.status

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    term_status = MOI.get(opt, MOI.TerminationStatus())
    if term_status == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif term_status == MOI.ALMOST_OPTIMAL
        return MOI.NEARLY_FEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function _adjust_obj(opt::Optimizer, value::Float64)
    sense_mul = (opt.obj_sense == MOI.MAX_SENSE ? -1 : 1)
    return sense_mul * (opt.obj_offset + value)
end

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return _adjust_obj(opt, opt.obj_value)
end

MOI.get(opt::Optimizer, ::MOI.ObjectiveBound) = _adjust_obj(opt, opt.obj_bound)

MOI.get(opt::Optimizer, ::MOI.RelativeGap) = get_obj_rel_gap(opt)

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(opt, attr)
    return opt.incumbent[vi.value]
end

function MOI.get(opt::Optimizer, ::MOI.NodeCount)
    return MOI.get(opt.oa_model, MOI.NodeCount())
end

"""
    NumberOfCuts

Number of cuts added to the outer approximation model.
"""
struct NumberOfCuts <: MOI.AbstractModelAttribute end

MOI.attribute_value_type(::NumberOfCuts) = Int64
MOI.is_set_by_optimize(::NumberOfCuts) = true

function MOI.get(opt::Optimizer, ::NumberOfCuts)
    return opt.num_cuts
end
