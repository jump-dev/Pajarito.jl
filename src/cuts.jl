# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# helpers for cuts

# add initial fixed cuts
function add_init_cuts(opt::Optimizer)
    num_cons = get_num_constraints(opt.oa_model)
    for cache in opt.cone_caches
        Cones.add_init_cuts(cache, opt)
    end
    opt.num_cuts += get_num_constraints(opt.oa_model) - num_cons
    return
end

# add relaxation dual cuts
function add_relax_cuts(opt::Optimizer)
    JuMP.has_duals(opt.relax_model) || return false

    cuts = JuMP.AffExpr[]
    for (ci, cache) in zip(opt.relax_oa_cones, opt.cone_caches)
        append!(cuts, get_dual_cuts(ci, cache, opt, false))
    end
    cuts_added = add_cuts(cuts, opt, false)

    if !cuts_added && opt.verbose
        println("continuous relaxation cuts could not be added")
    end
    return cuts_added
end

# add subproblem dual cuts
function add_subp_cuts(
    opt::Optimizer,
    viol_only::Bool,
    cuts_cache::Union{Nothing,Vector{AE}},
)
    JuMP.has_duals(opt.subp_model) || return false

    cuts = JuMP.AffExpr[]
    for (ci, cache) in zip(opt.subp_oa_cones, opt.cone_caches)
        append!(cuts, get_dual_cuts(ci, cache, opt, true))
    end
    cuts_added = add_cuts(cuts, opt, viol_only)

    if !isnothing(cuts_cache)
        append!(cuts_cache, cuts)
    end

    if !cuts_added && opt.verbose
        println("subproblem cuts could not be added")
    end
    return cuts_added
end

# add separation cuts (only if violated)
function add_sep_cuts(opt::Optimizer)
    s_vals = [
        get_value(Cones.get_oa_s(cache), opt.lazy_cb) for
        cache in opt.cone_caches
    ]

    cuts = JuMP.AffExpr[]
    for (s, cache) in zip(s_vals, opt.cone_caches)
        append!(cuts, Cones.get_sep_cuts(s, cache, opt))
    end
    cuts_added = add_cuts(cuts, opt, true)

    if !cuts_added && opt.verbose
        println("separation cuts could not be added")
    end
    return cuts_added
end

# get and rescale relaxation/subproblem dual
function get_dual_cuts(ci::CR, cache::Cache, opt::Optimizer, is_subp::Bool)
    z = JuMP.dual(ci)

    z_norm = LinearAlgebra.norm(z, Inf)
    if z_norm < 1e-10
        # discard duals with small norm
        return JuMP.AffExpr[]
    elseif z_norm > 1e11
        println("norm of dual is $z_norm")
    end

    # TODO subproblem-based cut rescaling like old Pajarito
    if is_subp
        status = JuMP.termination_status(opt.subp_model)
        if status in (MOI.INFEASIBLE, MOI.ALMOST_INFEASIBLE)
            # rescale dual rays
            z .*= inv(z_norm)
        end
    end

    return Cones.get_subp_cuts(z, cache, opt)
end
