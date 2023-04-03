# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=
positive semidefinite cone triangle (unscaled)
w ⪰ 0
=#

mutable struct PositiveSemidefiniteConeTriangle <: Cache
    oa_s::Vector{JuMP.AffExpr}
    d::Int
    W::Matrix{<:Union{JuMP.VariableRef,JuMP.AffExpr}}
    PositiveSemidefiniteConeTriangle() = new()
end

function create_cache(
    oa_s::Vector{JuMP.AffExpr},
    moi_cone::MOI.PositiveSemidefiniteConeTriangle,
    ::Optimizer,
)
    cache = PositiveSemidefiniteConeTriangle()
    cache.oa_s = oa_s
    cache.d = moi_cone.side_dimension
    cache.W = vec_to_symm(cache.d, oa_s)
    return cache
end

function add_init_cuts(cache::PositiveSemidefiniteConeTriangle, opt::Optimizer)
    # add variable bounds W_ii ≥ 0, ∀i
    d = cache.d
    W = cache.W
    JuMP.@constraint(opt.oa_model, [i in 1:d], W[i, i] >= 0)
    opt.use_init_fixed_oa || return

    # add cuts (1, 1, ±2) on (W_ii, W_jj, W_ij), ∀i != j
    # (a linearization of W_ii * W_jj ≥ W_ij^2)
    # initial OA polyhedron is the dual cone of diagonally dominant matrices
    for j in 1:d, i in 1:(j-1)
        Wii = W[i, i]
        Wjj = W[j, j]
        Wij = W[i, j]
        JuMP.@constraints(opt.oa_model, begin
            Wii + Wjj + 2 * Wij >= 0
            Wii + Wjj - 2 * Wij >= 0
        end)
    end
    return
end

function get_subp_cuts(
    z::Vector{Float64},
    cache::PositiveSemidefiniteConeTriangle,
    opt::Optimizer,
)
    # strengthened cuts from eigendecomposition are λᵢ * rᵢ * rᵢ'
    R = vec_to_symm(cache.d, z)
    F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(R, :U), 1e-9, Inf)
    isempty(F.values) && return JuMP.AffExpr[]
    R_eig = F.vectors * LinearAlgebra.Diagonal(sqrt.(F.values))
    return _get_cuts(R_eig, cache, opt)
end

function get_sep_cuts(
    s::Vector{Float64},
    cache::PositiveSemidefiniteConeTriangle,
    opt::Optimizer,
)
    # check s ∉ K
    sW = vec_to_symm(cache.d, s)
    F = LinearAlgebra.eigen!(
        LinearAlgebra.Symmetric(sW, :U),
        -Inf,
        -opt.tol_feas,
    )
    isempty(F.values) && return JuMP.AffExpr[]
    return _get_cuts(F.vectors, cache, opt)
end

function _get_cuts(
    R_eig::Matrix{Float64},
    cache::PositiveSemidefiniteConeTriangle,
    opt::Optimizer,
)
    # cuts from eigendecomposition are rᵢ * rᵢ'
    W = cache.W
    cuts = JuMP.AffExpr[]
    for i in 1:size(R_eig, 2)
        @views r_i = R_eig[:, i]
        R_i = r_i * r_i'
        clean_array!(R_i) && continue
        cut = dot_expr(R_i, W, opt)
        push!(cuts, cut)
    end
    return cuts
end

function vec_to_symm(d::Int, vec::Vector{T}) where {T}
    mat = Matrix{T}(undef, d, d)
    k = 1
    for j in 1:d
        for i in 1:(j-1)
            mat[i, j] = mat[j, i] = vec[k]
            k += 1
        end
        mat[j, j] = vec[k]
        k += 1
    end
    @assert k - 1 == length(vec)
    return mat
end
