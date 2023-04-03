# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=
power cone, parametrized by exponent t ∈ (0, 1)
(u, v, w) : u ≥ 0, v ≥ 0, u^t * v^(1-t) ≥ |w|
equivalently (v > 0, u ≥ v * |w / v|^(1/t))
dual cone
(p, q, r) : p ≥ 0, q ≥ 0, (p/t)^t * (q/(1-t))^(1-t) ≥ |r|
equivalently (q > 0, p ≥ t / (1-t) * q * |(1-t) * r / q|^(1/t))
=#

mutable struct PowerCone <: Cache
    t::Real
    oa_s::Vector{JuMP.AffineExpr}
    PowerCone() = new()
end

function create_cache(
    oa_s::Vector{JuMP.AffineExpr},
    moi_cone::MOI.PowerCone,
    ::Optimizer,
)
    @assert length(oa_s) == 3
    cache = PowerCone()
    cache.t = moi_cone.exponent
    cache.oa_s = oa_s
    return cache
end

function add_init_cuts(cache::PowerCone, opt::Optimizer)
    # add variable bounds
    (u, v, w) = cache.oa_s
    JuMP.@constraint(opt.oa_model, u >= 0)
    JuMP.@constraint(opt.oa_model, v >= 0)
    opt.use_init_fixed_oa || return

    # add cuts (t, 1-t, ±1)
    t = cache.t
    JuMP.@constraints(opt.oa_model, begin
        t * u + (1 - t) * v + w >= 0
        t * u + (1 - t) * v - w >= 0
    end)
    return
end

function get_subp_cuts(z::Vector{Float64}, cache::PowerCone, opt::Optimizer)
    p = z[1]
    q = z[2]
    if min(p, q) < 0
        # z ∉ K
        @warn("dual vector is not in the dual cone")
        return JuMP.AffineExpr[]
    end

    # strengthened cut is (p, q, sign(r) * (p/t)^t * (q/(1-t))^(1-t))
    t = cache.t
    r = sign(z[3]) * (p / t)^t * (q / (1 - t))^(1 - t)
    (u, v, w) = cache.oa_s
    cut = JuMP.@expression(opt.oa_model, p * u + q * v + r * w)
    return [cut]
end

function get_sep_cuts(s::Vector{Float64}, cache::PowerCone, opt::Optimizer)
    us = max(s[1], 0)
    vs = max(s[2], 0)
    ws = s[3]

    # check s ∉ K
    t = cache.t
    if us >= 0 && vs >= 0 && (us^t * vs^(1 - t) - abs(ws)) > -opt.tol_feas
        return JuMP.AffineExpr[]
    end

    # gradient cut is (t * (us/vs)^(t-1), (1-t) * (us/vs)^t, -sign(ws))
    # perturb point if u or v is near zero
    us = max(us, 1e-8)
    vs = max(vs, 1e-8)
    (u, v, w) = cache.oa_s
    p = t * (us / vs)^(t - 1)
    q = (1 - t) * (us / vs)^t
    cut = JuMP.@expression(opt.oa_model, p * u + q * v - sign(ws) * w)
    return [cut]
end
