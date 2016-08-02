#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runconicdefaulttests(mip_solver_drives)
    facts("Default solvers test") do
        context("With $(mip_solver_drives ? "BC" : "OA"), defaulting to $(typeof(MathProgBase.defaultMIPsolver)) and $(typeof(MathProgBase.defaultConicsolver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize(3x,
                                x <= 10,
                                x^2 <= 9)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives,log_level=0))

            @fact problem.optval --> roughly(9.0, TOL)
        end
    end
end

function runconictests(mip_solver_drives, mip_solver, conic_solver)
    algorithm = mip_solver_drives ? "BC" : "OA"

    facts("Infeasible conic problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize(3x,
                                x >= 4,
                                x^2 <= 9)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

            @fact problem.status --> :Infeasible
        end
    end

    facts("Univariate maximization problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)

            problem = Convex.maximize(3x,
                                x <= 10,
                                x^2 <= 9)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

            @fact problem.optval --> roughly(9.0, TOL)
        end
    end

    facts("Continuous problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1)
            y = Convex.Variable(1, Convex.Positive())

            problem = Convex.maximize(3x + y,
                                x >= 0,
                                3x + 2y <= 10,
                                exp(x) <= 10)

           @fact_throws ErrorException Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))
       end
    end

    facts("Maximization problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1, Convex.Positive())

            problem = Convex.maximize(3x + y,
                                x >= 0,
                                3x + 2y <= 10,
                                exp(x) <= 10)

           Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

           @fact problem.optval --> roughly(8.0, TOL)
       end
    end

    facts("Solver test") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1)

            problem = Convex.minimize(-3x - y,
                               x >= 1,
                               y >= 0,
                               3x + 2y <= 10,
                               x^2 <= 5,
                               exp(y) + x <= 7)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

            # @fact problem.status --> :Optimal
            @fact Convex.evaluate(x) --> roughly(2.0, TOL)
        end
    end

    facts("No SOC disaggregation test") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1)

            problem = Convex.minimize(-3x - y,
                               x >= 1,
                               y >= 0,
                               3x + 2y <= 10,
                               x^2 <= 5,
                               exp(y) + x <= 7)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, disagg_soc=false, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

            # @fact problem.status --> :Optimal
            @fact Convex.evaluate(x) --> roughly(2.0, TOL)
        end
    end

    facts("Solver test 2") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
            x = Convex.Variable(1,:Int)
            y = Convex.Variable(1, Convex.Positive())

            problem = Convex.minimize(-3x - y,
                               x >= 1,
                               3x + 2y <= 30,
                               exp(y^2) + x <= 7)

            Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver,log_level=0))

            # @fact problem.status --> :Optimal
            @fact Convex.evaluate(x) --> roughly(6.0, TOL)
        end
    end
end
