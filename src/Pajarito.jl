# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# a solver for mixed-integer conic problems using outer approximation and conic duality

module Pajarito

import Printf
import LinearAlgebra
import SparseArrays

import JuMP
const MOI = JuMP.MOI
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction{Float64}
const SOS12 = Union{MOI.SOS1{Float64},MOI.SOS2{Float64}}
const VR = JuMP.VariableRef
const CR = JuMP.ConstraintRef
const AE = JuMP.AffExpr

abstract type Cache end

include("optimizer.jl")
include("Cones/Cones.jl")
include("algorithms.jl")
include("models.jl")
include("cuts.jl")
include("JuMP_tools.jl")
include("MOI_wrapper.jl")
include("MOI_copy_to.jl")

end
