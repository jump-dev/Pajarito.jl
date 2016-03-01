import Convex
using Pajarito
using FactCheck
using Compat


facts("Univariate maximization problem") do
    x = Convex.Variable(1,:Int)

    problem = Convex.maximize( 3x,
                        x <= 10,
                        x^2 <= 9)

    Convex.solve!(problem, PajaritoDCPSolver(dcp_solver=ConicNLPSolver()))
    
    @fact problem.optval --> roughly(9.0)

end


facts("Maximization problem") do
    x = Convex.Variable(1,:Int)
    y = Convex.Variable(1, Convex.Positive())

    problem = Convex.maximize( 3x+y,
                        x >= 0,
                        3x + 2y <= 10,
                        exp(x) <= 20)

    Convex.solve!(problem, PajaritoDCPSolver(verbose=2,dcp_solver=ConicNLPSolver()))
    
    @fact problem.optval --> roughly(8.0)

end

facts("Solver test") do

    x = Convex.Variable(1,:Int)
    y = Convex.Variable(1)

    problem = Convex.minimize(-3x-y,
                       x >= 1,
                       y >= 0,
                       3x + 2y <= 10,
                       x^2 <= 5,
                       exp(y) + x <= 7)

    Convex.solve!(problem, PajaritoDCPSolver(verbose=2,dcp_solver=ConicNLPSolver())) 

    @fact problem.status --> :Optimal
    @fact Convex.evaluate(x) --> roughly(2.0)
end

facts("Solver test 2") do

    x = Convex.Variable(1,:Int)
    y = Convex.Variable(1, Convex.Positive())

    problem = Convex.minimize(-3x-y,
                       x >= 1,
                       3x + 2y <= 30,
                       exp(y^2) + x <= 7)

    Convex.solve!(problem, PajaritoDCPSolver(dcp_solver=ConicNLPSolver()))

    @fact problem.status --> :Optimal
    @fact Convex.evaluate(x) --> roughly(6.0)
end
