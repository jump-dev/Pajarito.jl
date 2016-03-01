using JuMP
using Pajarito
using FactCheck
using Compat

include(Pkg.dir("JuMP","test","solvers.jl"))

include("dcptest.jl")

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

facts("Convex constraint with LB and UB") do
    m = Model(solver=PajaritoSolver())

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)

    @addNLConstraint(m, 8 <= x^2 <= 10)

    @fact_throws ErrorException solve(m)

end

facts("Infeasible NLP problem") do
    m = Model(solver=PajaritoSolver())

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)

    @addNLConstraint(m, x^2 >= 9)
    @addNLConstraint(m, exp(y) + x <= 2)

    status = solve(m)

    @fact status --> :Infeasible
end

facts("Infeasible MIP problem") do
    m = Model(solver=PajaritoSolver())

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)
    @addConstraint(m, 6x + 5y >= 30)

    @addNLConstraint(m, x^2 >= 8)
    @addNLConstraint(m, exp(y) + x <= 7)

    status = solve(m)

    @fact status --> :Infeasible
end

facts("Solver test") do
for ip_solver in ip_solvers
for nlp_solver in convex_nlp_solvers
    string(typeof(nlp_solver)) == "NLoptSolver" && continue
    string(typeof(nlp_solver)) == "MosekSolver" && continue
    string(typeof(nlp_solver)) == "OsilSolver" && continue
    string(typeof(ip_solver)) == "OsilSolver" && continue
    string(typeof(ip_solver)) == "CbcSolver" && continue
context("With $(typeof(ip_solver)) and $(typeof(nlp_solver))") do

    m = Model(solver=PajaritoSolver(verbose=0,mip_solver=ip_solver,nlp_solver=nlp_solver))

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)
    @addConstraint(m, x >= 1)

    @addNLConstraint(m, x^2 <= 5)
    @addNLConstraint(m, exp(y) + x <= 7)

    status = solve(m)

    @fact status --> :Optimal
    @fact getValue(x) --> 2.0
end; end; end
end

facts("Optimal solution with nonlinear objective") do
    m = Model(solver=PajaritoSolver(verbose=0))

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setNLObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)
    @addConstraint(m, x >= 1)

    @addNLConstraint(m, x^2 <= 5)
    @addNLConstraint(m, exp(y) + x <= 7)

    status = solve(m)

    @fact status --> :Optimal
    @fact getValue(x) --> 2.0
end

# TODO setvartype is not called if there are no integer variables in the model
facts("No integer variables") do
    m = Model(solver=PajaritoSolver(verbose=0))

    @defVar(m, x >= 0, start = 1)
    @defVar(m, y >= 0, start = 1)

    @setNLObjective(m, Min, -3x - y)

    @addConstraint(m, 3x + 2y + 10 <= 20)
    @addConstraint(m, x >= 1)

    @addNLConstraint(m, x^2 <= 5)
    @addNLConstraint(m, exp(y) + x <= 7)

    status = solve(m)

    @fact status --> :Optimal
    # TODO CHECK SOLUTION APPROXIMATELY
end

facts("Maximization problem") do
    m = Model(solver=PajaritoSolver())

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Max, 3x + y)

    @addConstraint(m, 3x + 2y + 10 <= 20)

    @addNLConstraint(m, x^2 <= 9)

    status = solve(m)
    @fact round(getObjectiveValue(m)-9.5) --> 0.0

end

facts("Maximization problem with nonlinear function") do
    m = Model(solver=PajaritoSolver())

    @defVar(m, x >= 0, start = 1, Int)
    @defVar(m, y >= 0, start = 1)

    @setObjective(m, Max, -x^2 - y)

    @addConstraint(m, x + 2y >= 4)

    @addNLConstraint(m, x^2 <= 9)

    status = solve(m)
    @fact round(getObjectiveValue(m)-2.0) --> 0.0

end
