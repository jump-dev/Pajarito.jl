facts("Empty MIP solver test") do
    x = Convex.Variable(1,:Int)

    problem = Convex.maximize( 3x,
                        x <= 10,
                        x^2 <= 9)

    @fact_throws Convex.solve!(problem, PajaritoConicSolver(conic_solver=ConicNLPWrapper(nlp_solver=nlp_solver))) "MIP solver is not specified."

end

facts("Empty DCP solver test") do
    x = Convex.Variable(1,:Int)

    problem = Convex.maximize( 3x,
                        x <= 10,
                        x^2 <= 9)

    @fact_throws Convex.solve!(problem, PajaritoConicSolver(mip_solver=mip_solver)) "DCP solver is not specified."

end

facts("Univariate maximization problem") do
    x = Convex.Variable(1,:Int)

    problem = Convex.maximize( 3x,
                        x <= 10,
                        x^2 <= 9)

    for i = 1:length(algorithms)
        Convex.solve!(problem, PajaritoConicSolver(algorithm=algorithms[i],mip_solver=mip_solver,conic_solver=ConicNLPWrapper(nlp_solver=nlp_solver)))
        
        @fact problem.optval --> roughly(9.0)
    end
end


facts("Maximization problem") do
    x = Convex.Variable(1,:Int)
    y = Convex.Variable(1, Convex.Positive())

    problem = Convex.maximize( 3x+y,
                        x >= 0,
                        3x + 2y <= 10,
                        exp(x) <= 20)

    for i = 1:length(algorithms)
        Convex.solve!(problem, PajaritoConicSolver(algorithm=algorithms[i],mip_solver=mip_solver,conic_solver=ConicNLPWrapper(nlp_solver=nlp_solver)))
        
        @fact problem.optval --> roughly(8.0)
    end

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

    for i = 1:length(algorithms)
        Convex.solve!(problem, PajaritoConicSolver(algorithm=algorithms[i],mip_solver=mip_solver,conic_solver=ConicNLPWrapper(nlp_solver=nlp_solver))) 

        @fact problem.status --> :Optimal
        @fact Convex.evaluate(x) --> roughly(2.0)
    end
end

facts("Solver test 2") do

    x = Convex.Variable(1,:Int)
    y = Convex.Variable(1, Convex.Positive())

    problem = Convex.minimize(-3x-y,
                       x >= 1,
                       3x + 2y <= 30,
                       exp(y^2) + x <= 7)

    for i = 1:length(algorithms)
        Convex.solve!(problem, PajaritoConicSolver(algorithm=algorithms[i],mip_solver=mip_solver,conic_solver=ConicNLPWrapper(nlp_solver=nlp_solver)))

        @fact problem.status --> :Optimal
        @fact Convex.evaluate(x) --> roughly(6.0)
    end
end
