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

module Pajarito

import MathProgBase

using Printf
using SparseArrays
using LinearAlgebra

using ConicBenchmarkUtilities # to bring in SparseArrays.findnz impl

include("conic_dual_solver.jl")
include("solver.jl")
include("conic_algorithm.jl")

end
