#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using FactCheck
using JuMP
import Convex
using Pajarito

include(Pkg.dir("JuMP", "test", "solvers.jl"))

# Define solvers using JuMP/test/solvers.jl
# solvers_mip = lazy_solvers
using Gurobi
solvers_mip = [GurobiSolver(OutputFlag=1)]
solvers_nlnr = []
ipt && push!(solvers_nlnr, Ipopt.IpoptSolver(print_level=0))
kni && push!(solvers_nlnr, KNITRO.KnitroSolver(objrange=1e16,outlev=0,maxit=100000))
solvers_conic = eco ? [ECOS.ECOSSolver(verbose=false)] : []
solvers_sdp = mos ? [Mosek.MosekSolver(LOG=0)] : []

@show solvers_mip
@show solvers_nlnr
@show solvers_conic
@show solvers_sdp

# Set fact check tolerance
TOL = 1e-3

# Nonlinear models tests in nlptest.jl
include("nlptest.jl")
for mip_solver_drives in [false, true], mip in solvers_mip, nlnr in solvers_nlnr
    runnonlineartests(mip_solver_drives, mip, nlnr)
end

# Conic models test in conictest.jl
include("conictest.jl")

# Default solvers test
runconicdefaulttests(false)

log = 0
for mip_solver_drives in [false, true], mip in solvers_mip
    # Conic model with conic solvers
    for conic in solvers_conic
        runconictests(mip_solver_drives, mip, conic, log)
    end

    # Conic model with nonlinear solvers
    for nlnr in solvers_nlnr
        runconictests(mip_solver_drives, mip, nlnr, log)
    end
end

# SDP conic models tests in sdptest.jl
include("sdptest.jl")
for mip_solver_drives in [false], mip in solvers_mip, sdp in solvers_sdp
    runsdptests(mip_solver_drives, mip, sdp)
end

FactCheck.exitstatus()
