#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

#=========================================================
This package contains the mixed-integer non-linear programming
(MINLP) solver Pajarito. It applies outer approximation to a
sequence of mixed-integer linear (or second-order cone) programming
problems that approximate the original MINLP, until convergence.
=========================================================#

__precompile__()


module Pajarito
    import MathProgBase
    using ConicNonlinearBridge

    include("solver.jl")
    include("conic_algorithm.jl")
    include("nonlinear_algorithm.jl")
end
