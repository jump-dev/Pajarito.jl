#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using FactCheck
using JuMP
import Convex
using Pajarito
using GLPKMathProgInterface
using ECOS
using Ipopt
using Mosek

include(Pkg.dir("JuMP", "test", "solvers.jl"))

# Define solvers
mip_solvers = [GLPKSolverMIP()]
nlp_solvers = [IpoptSolver(print_level=0)]
conic_solvers = [ECOSSolver(verbose=0)]
sdp_solvers = [MosekSolver(LOG=0)]

# Set fact check tolerance
TOL = 1e-3

# Nonlinear models tests in nlptest.jl
include("nlptest.jl")
for branch_cut in [true, false], mip_solver in mip_solvers, nlp_solver in nlp_solvers
    runnonlineartests(branch_cut, mip_solver, nlp_solver)
end

# Conic models test in conictest.jl
include("conictest.jl")
# Conic model with conic solver
for branch_cut in [true, false], mip_solver in mip_solvers, conic_solver in conic_solvers
    runconictests(branch_cut, mip_solver, conic_solver)
end
# Conic model with nonlinear solver
for branch_cut in [true, false], mip_solver in mip_solvers, nlp_solver in nlp_solvers
    runconictests(branch_cut, mip_solver, nlp_solver)
end

# SDP conic models tests in sdptest.jl
include("sdptest.jl")
for branch_cut in [true, false], mip_solver in mip_solvers, sdp_solver in sdp_solvers
    runsdptests(branch_cut, mip_solver, sdp_solver)
end

FactCheck.exitstatus()
