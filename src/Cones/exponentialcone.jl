# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=
exponential cone
(u, v, w) : (u ≤ 0, v = 0, w ≥ 0) or (v > 0, w ≥ v * exp(u / v))
equivalently (v > 0, u ≤ v * log(w / v))
dual cone
(p, q, r) : (p = 0, q ≥ 0, r ≥ 0) or (p < 0, r ≥ -p * exp(q / p - 1))
equivalently (p < 0, q ≥ p * (log(r / -p) + 1))
=#

mutable struct ExponentialCone <: Cache
    oa_s::Vector{AE}
    ExponentialCone() = new()
end

function create_cache(oa_s::Vector{AE}, ::MOI.ExponentialCone, ::Optimizer)
    @assert length(oa_s) == 3
    cache = ExponentialCone()
    cache.oa_s = oa_s
    return cache
end

function add_init_cuts(cache::ExponentialCone, opt::Optimizer)
    # add variable bounds
    (u, v, w) = cache.oa_s
    JuMP.@constraint(opt.oa_model, v >= 0)
    JuMP.@constraint(opt.oa_model, w >= 0)
    opt.use_init_fixed_oa || return

    # add cuts from linearizations at p = -1
    r_lin = [1e-3, 1e0, 1e2]
    JuMP.@constraint(
        opt.oa_model,
        [r in r_lin],
        -u - (log(r) + 1) * v + r * w >= 0
    )
    return
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::ExponentialCone,
    opt::Optimizer,
)
    p = z[1]
    r = z[3]
    if p > 0 || r < 0
        # z ∉ K
        @warn("exponential cone dual vector violates variable bounds")
        return AE[]
    end
    q = z[2]
    (u, v, w) = cache.oa_s

    if p > -1e-12
        return AE[]
    elseif r / -p > 1e-8
        # strengthened cut is (p, p * (log(r / -p) + 1), r)
        q = p * (log(r / -p) + 1)
        cut = JuMP.@expression(opt.oa_model, p * u + q * v + r * w)
    elseif q / p < 30
        # strengthened cut is (p, q, -p * exp(q / p - 1))
        r = -p * exp(q / p - 1)
        cut = JuMP.@expression(opt.oa_model, p * u + q * v + r * w)
    else
        return AE[]
    end
    return [cut]
end

function get_sep_cuts(
    s::Vector{Float64},
    cache::ExponentialCone,
    opt::Optimizer,
)
    us = s[1]
    vs = max(s[2], 0)
    ws = max(s[3], 0)
    (u, v, w) = cache.oa_s

    # check s ∉ K and add cut
    if vs <= 1e-7
        # vs near zero, so violation is us
        if us >= opt.tol_feas
            if ws <= 1e-12
                @warn("cannot add separation cut for exponential cone")
            end

            # cut is (-1, -log(ℯ / 2 * us / ws), us / ws / 2)
            r = us / (2 * ws)
            q = -log(ℯ / 2 * r)
            cut = JuMP.@expression(opt.oa_model, -u + q * v + r * w)
        else
            return AE[]
        end
    elseif ws / vs > 1e-8 && us - vs * log(ws / vs) > opt.tol_feas
        # vs and ws not near zero
        # gradient cut is (-1, log(ws / vs) - 1, vs / ws)
        q = log(ws / vs) - 1
        cut = JuMP.@expression(opt.oa_model, -u + q * v + vs / ws * w)
    elseif vs * exp(us / vs) - ws > opt.tol_feas
        # gradient cut is (-exp(us / vs), (us - vs) / vs * exp(us / vs), 1)
        p = -exp(us / vs)
        q = (us - vs) / vs * -p
        cut = JuMP.@expression(opt.oa_model, p * u + q * v + w)
    else
        return AE[]
    end
    return [cut]
end
