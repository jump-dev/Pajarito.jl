#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This package contains the mixed-integer convex programming (MICP)
solver Pajarito. It applies outer approximation to a sequence
of mixed-integer linear (or second-order cone) programming
problems that approximate the original MICP, until convergence.
=========================================================#

__precompile__()


module Pajarito

import MathProgBase

using Compat.Printf
using Compat.SparseArrays
using Compat.LinearAlgebra

import Compat: undef
import Compat: @warn
import Compat: stdout
import Compat: stderr
import Compat: findall
import Compat: hasmethod
import Compat: rmul!
import Compat: norm

# Needed for only for Julia v0.6 compatability
if VERSION < v"0.7.0-"
    eigen! = eigfact!
end

if VERSION > v"0.7.0-"
    # this is required because findall return type changed in v0.7
    function SparseArrays.findnz(A::AbstractMatrix)
        I = findall(!iszero, A)
        return (getindex.(I, 1), getindex.(I, 2), A[I])
    end
end

include("conic_dual_solver.jl")
include("solver.jl")
include("conic_algorithm.jl")

end
