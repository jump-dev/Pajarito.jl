using ConicNonlinearBridge
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
dcp_solver=ConicNLPWrapper(nlp_solver=nlp_solver)

include(Pkg.dir("JuMP","test","solvers.jl"))

TOL = 1e-3

include("nlptest.jl")
include("conictest.jl")

for i = 1:length(ip_solvers)
    runconictests("OA", ip_solvers[i], dcp_solver)
    runconictests("OA", ip_solvers[i], conic_solver)
end

for i = 1:length(lazy_solvers)
    contains(string(typeof(lazy_solvers[i])),"GLPKSolverMIP") && continue
    runconictests("BC", lazy_solvers[i], dcp_solver)
    runconictests("BC", lazy_solvers[i], conic_solver)
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
