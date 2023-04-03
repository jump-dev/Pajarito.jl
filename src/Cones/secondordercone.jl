# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=
second-order cone
(u, w) : u ≥ ‖w‖

extended formulation
(u, w) : u ≥ 0, ∃ ϕ ≥ 0, u ≥ 2 Σᵢ ϕᵢ, 2 u ϕᵢ ≥ wᵢ²
linear and 3-dim rotated second order cone constraints
=#

mutable struct SecondOrderCone{E<:NatExt} <: Cache
    oa_s::Vector{JuMP.AffineExpr}
    d::Int
    ϕ::Vector{JuMP.VariableRef}
    SecondOrderCone{E}() where {E<:NatExt} = new{E}()
end

function create_cache(
    oa_s::Vector{JuMP.AffineExpr},
    moi_cone::MOI.SecondOrderCone,
    opt::Optimizer,
)
    dim = MOI.dimension(moi_cone)
    @assert dim == length(oa_s)
    d = dim - 1
    E = nat_or_ext(opt, d)
    cache = SecondOrderCone{E}()
    cache.oa_s = oa_s
    cache.d = d
    return cache
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::SecondOrderCone,
    opt::Optimizer,
)
    return _get_cuts(z[2:end], cache, opt)
end

function get_sep_cuts(
    s::Vector{Float64},
    cache::SecondOrderCone,
    opt::Optimizer,
)
    us = s[1]
    @views ws = s[2:end]
    ws_norm = LinearAlgebra.norm(ws)
    # check s ∉ K
    if us - ws_norm > -opt.tol_feas
        return JuMP.AffineExpr[]
    end

    # cut is (1, -ws / ‖ws‖)
    r = ws / -ws_norm
    return _get_cuts(r, cache, opt)
end

# unextended formulation

function add_init_cuts(cache::SecondOrderCone{Nat}, opt::Optimizer)
    # add variable bound
    u = cache.oa_s[1]
    JuMP.@constraint(opt.oa_model, u >= 0)
    opt.use_init_fixed_oa || return

    # add cuts u ≥ |wᵢ|
    @views w = cache.oa_s[2:end]
    d = cache.d
    JuMP.@constraints(opt.oa_model, begin
        [i in 1:d], u >= w[i]
        [i in 1:d], u >= -w[i]
    end)
    return
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderCone{Nat},
    opt::Optimizer,
)
    # strengthened cut is (‖r‖, r)
    clean_array!(r) && return JuMP.AffineExpr[]
    p = LinearAlgebra.norm(r)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end]
    cut = JuMP.@expression(opt.oa_model, p * u + LinearAlgebra.dot(r, w))
    return [cut]
end

# extended formulation

num_ext_variables(cache::SecondOrderCone{Ext}) = cache.d

function extend_start(
    cache::SecondOrderCone{Ext},
    s_start::Vector{Float64},
    opt::Optimizer,
)
    u_start = s_start[1]
    w_start = s_start[2:end]
    if u_start < opt.tol_feas
        return zeros(cache.d)
    end
    return [w_i / 2u_start * w_i for w_i in w_start]
end

function setup_auxiliary(cache::SecondOrderCone{Ext}, opt::Optimizer)
    @assert cache.d >= 2
    ϕ = cache.ϕ = JuMP.@variable(opt.oa_model, [1:(cache.d)], lower_bound = 0)
    u = cache.oa_s[1]
    JuMP.@constraint(opt.oa_model, u >= 2 * sum(ϕ))
    return ϕ
end

function add_init_cuts(cache::SecondOrderCone{Ext}, opt::Optimizer)
    # add variable bound
    u = cache.oa_s[1]
    JuMP.@constraint(opt.oa_model, u >= 0)
    opt.use_init_fixed_oa || return

    # add disaggregated cuts (1, 2, ±2) on (u, ϕᵢ, wᵢ), implying u ≥ |wᵢ|
    @views w = cache.oa_s[2:end]
    d = cache.d
    ϕ = cache.ϕ
    JuMP.@constraints(
        opt.oa_model,
        begin
            [i in 1:d], u + 2 * ϕ[i] + 2 * w[i] >= 0
            [i in 1:d], u + 2 * ϕ[i] - 2 * w[i] >= 0
        end
    )
    return
end

function _get_cuts(
    r::Vector{Float64},
    cache::SecondOrderCone{Ext},
    opt::Optimizer,
)
    clean_array!(r) && return JuMP.AffineExpr[]
    p = LinearAlgebra.norm(r)
    u = cache.oa_s[1]
    @views w = cache.oa_s[2:end]
    ϕ = cache.ϕ
    cuts = JuMP.AffineExpr[]
    for i in 1:(cache.d)
        r_i = r[i]
        iszero(r_i) && continue
        # strengthened disaggregated cut on (u, ϕᵢ, wᵢ) is (rᵢ² / 2‖r‖, ‖r‖, rᵢ)
        cut = JuMP.@expression(
            opt.oa_model,
            r_i^2 / 2p * u + p * ϕ[i] + r_i * w[i]
        )
        push!(cuts, cut)
    end
    return cuts
end
