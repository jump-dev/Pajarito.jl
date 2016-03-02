using JuMP
using Pajarito
using FactCheck
using CPLEX
using SCS
using Ipopt
import Convex

# Define solvers
mip_solver = CplexSolver(CPX_PARAM_SCRIND=0,CPX_PARAM_REDUCE=0,CPX_PARAM_EPINT=1e-8,CPX_PARAM_EPRHS=1e-8)
nlp_solver = IpoptSolver(print_level=0)
conic_solver = SCSSolver(verbose=0)

algorithms = ["P-OA", "P-CB"]

include(Pkg.dir("JuMP","test","solvers.jl"))

TOL = 1e-3

include("nlptest.jl")
include("dcptest.jl")
include("conictest.jl")

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


