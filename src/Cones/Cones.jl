# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# cut oracles for MathOptInterface cones

module Cones

import LinearAlgebra
import JuMP
const MOI = JuMP.MOI
const VR = JuMP.VariableRef
const AE = JuMP.AffExpr

import Pajarito: Cache, Optimizer

abstract type NatExt end
struct Nat <: NatExt end
struct Ext <: NatExt end
function nat_or_ext(opt::Optimizer, d::Int)
    return ((d > 1 && opt.use_extended_form) ? Ext : Nat)
end

include("secondordercone.jl")
include("exponentialcone.jl")
include("powercone.jl")
include("positivesemidefiniteconetriangle.jl")

# supported cones for outer approximation
const OACone = Union{
    MOI.SecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{Float64},
    MOI.PositiveSemidefiniteConeTriangle,
}

setup_auxiliary(::Cache, ::Optimizer) = VR[]

extend_start(::Cache, ::Vector{Float64}, ::Optimizer) = Float64[]

num_ext_variables(::Cache) = 0

function dot_expr(
    z::AbstractVecOrMat{Float64},
    vars::AbstractVecOrMat{<:Union{VR,AE}},
    opt::Optimizer,
)
    return JuMP.@expression(opt.oa_model, JuMP.dot(z, vars))
end

function clean_array!(z::AbstractArray)
    # avoid poorly conditioned cuts and near-zero values
    z_norm = LinearAlgebra.norm(z, Inf)
    min_abs = max(1e-12, 1e-15 * z_norm) # TODO tune/option
    for (i, z_i) in enumerate(z)
        if abs(z_i) < min_abs
            z[i] = 0
        end
    end
    return iszero(z)
end

get_oa_s(cache::Cache) = cache.oa_s

end
