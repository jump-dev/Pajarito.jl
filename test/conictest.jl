#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runconictests(algorithm, mip_solvers, conic_solvers)

facts("Default solvers test") do
    x = Convex.Variable(1,:Int)

    problem = Convex.maximize( 3x,
                        x <= 10,
                        x^2 <= 9)
    
    Convex.solve!(problem, PajaritoSolver())
    @fact problem.optval --> roughly(9.0, TOL)
end

facts("Default MIP solver test") do
    for conic_solver in conic_solvers
context("With $algorithm, $(typeof(conic_solver))") do
        x = Convex.Variable(1,:Int)

        problem = Convex.maximize( 3x,
                            x <= 10,
                            x^2 <= 9)
        
        Convex.solve!(problem, PajaritoSolver(cont_solver=conic_solver))
        @fact problem.optval --> roughly(9.0, TOL)
end
    end
end

facts("Default DCP solver test") do
    for mip_solver in mip_solvers
context("With $algorithm, $(typeof(mip_solver))") do
        x = Convex.Variable(1,:Int)

        problem = Convex.maximize( 3x,
                            x <= 10,
                            x^2 <= 9)
        
        Convex.solve!(problem, PajaritoSolver(mip_solver=mip_solver))
        @fact problem.optval --> roughly(9.0, TOL)
end
    end
end

facts("Univariate maximization problem") do

    for mip_solver in mip_solvers
        for conic_solver in conic_solvers
context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize( 3x,
                                x <= 10,
                                x^2 <= 9)
            Convex.solve!(problem, PajaritoSolver(algorithm=algorithm,mip_solver=mip_solver,cont_solver=conic_solver))
            @fact problem.optval --> roughly(9.0, TOL)
end
        end
    end

end


facts("Maximization problem") do
    for mip_solver in mip_solvers
        for conic_solver in conic_solvers
context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1, Convex.Positive())

            problem = Convex.maximize( 3x+y,
                                x >= 0,
                                3x + 2y <= 10,
                                exp(x) <= 10)

           Convex.solve!(problem, PajaritoSolver(algorithm=algorithm,mip_solver=mip_solver,cont_solver=conic_solver)) 
           @fact problem.optval --> roughly(8.0, TOL)
end
        end
    end

end

facts("Solver test") do

    for mip_solver in mip_solvers
        for conic_solver in conic_solvers
context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1)

            problem = Convex.minimize(-3x-y,
                               x >= 1,
                               y >= 0,
                               3x + 2y <= 10,
                               x^2 <= 5,
                               exp(y) + x <= 7)


            Convex.solve!(problem, PajaritoSolver(algorithm=algorithm,mip_solver=mip_solver,cont_solver=conic_solver)) 

            @fact problem.status --> :Optimal
            @fact Convex.evaluate(x) --> roughly(2.0, TOL)
end
        end
    end

end

facts("Solver test 2") do

    for mip_solver in mip_solvers
        for conic_solver in conic_solvers
context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do

            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1, Convex.Positive())

            problem = Convex.minimize(-3x-y,
                               x >= 1,
                               3x + 2y <= 30,
                               exp(y^2) + x <= 7)

            Convex.solve!(problem, PajaritoSolver(algorithm=algorithm,mip_solver=mip_solver,cont_solver=conic_solver))

            @fact problem.status --> :Optimal
            @fact Convex.evaluate(x) --> roughly(6.0, TOL)
end
        end
    end

end


if algorithm == "OA"
facts("Print test") do

    for mip_solver in mip_solvers
        for conic_solver in conic_solvers
context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1)

            problem = Convex.minimize(-3x-y,
                               x >= 1,
                               y >= 0,
                               3x + 2y <= 10,
                               x^2 <= 5,
                               exp(y) + x <= 7)

            Convex.solve!(problem, PajaritoSolver(verbose=1,algorithm=algorithm,mip_solver=mip_solver,cont_solver=conic_solver))

            @fact problem.status --> :Optimal

end
        end
    end

end
end

end
