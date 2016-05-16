#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
using Pajarito
using FactCheck
using GLPKMathProgInterface
using ECOS
using Ipopt
import Convex

# Define solvers
mip_solver = GLPKSolverMIP()
nlp_solver = IpoptSolver(print_level=0)
conic_solver = ECOSSolver(verbose=0)
myconic_solvers = [conic_solver, nlp_solver]
algorithms = ["OA", "BC"]

include(Pkg.dir("JuMP","test","solvers.jl"))

# TODO remove the following 
myip_solvers = Any[]
for i = 1:length(ip_solvers)
    contains(string(typeof(ip_solvers[i])), "MosekSolver") && continue 
    push!(myip_solvers, ip_solvers[i])
end

TOL = 1e-3

include("nlptest.jl")
include("conictest.jl")
#include("sdptest.jl")

runnonlineartests("OA", myip_solvers)
runnonlineartests("BC", lazy_solvers)

runconictests("OA", myip_solvers, myconic_solvers)
runconictests("BC", lazy_solvers, myconic_solvers)

for i = 1:length(sdp_solvers)
    if contains(string(typeof(sdp_solvers[i])), "MosekSolver") 
        #runsdptests("OA", myip_solvers, sdp_solvers[i])
        #runsdptests("BC", lazy_solvers, sdp_solvers[i])
        runSOCRotatedtests("OA", myip_solvers, sdp_solvers[i])
        runSOCRotatedtests("BC", lazy_solvers, sdp_solvers[i])
    end
end

# PAJARITO UNIT-TESTS
# 1. CONVEX CONSTRAINT WITH LB AND UB
# 2. INFEASIBLE NLP PROBLEM
# 3. INFEASIBLE MIP PROBLEM
# 4. OPTIMALITY TEST
#    a. SOLVER TEST
#    b. NL OBJECTIVE TEST
# 5. TODO NO INTEGER VARIABLES
# 6. TODO VARIABLES WITHOUT BOUNDS 
# 7. MAXIMIZATION PROBLEM 
# 8. MAXIMIZATION PROBLEM WITH NONLINEAR OBJECTIVE 

FactCheck.exitstatus()
