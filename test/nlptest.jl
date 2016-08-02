#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runnonlineartests(mip_solver_drives, mip_solver, nlp_solver)
    algorithm = mip_solver_drives ? "BC" : "OA"

    facts("Sparse matrix bug test") do
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @NLconstraint(m, y^2 <= 10)

        @fact solve(m) --> :Optimal
    end

    facts("Convex constraint with LB and UB test") do
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        @fact_throws ErrorException solve(m)
    end

    facts("Infeasible NLP problem") do
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 >= 9)
        @NLconstraint(m, exp(y) + x <= 2)

        status = solve(m)

        @fact status --> :Infeasible
    end

    facts("Infeasible MIP problem") do
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, 6x + 5y >= 30)
        @NLconstraint(m, x^2 >= 8)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve(m)

        @fact status --> :Infeasible
    end

    facts("Solver test") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

            @variable(m, x >= 0, start = 1, Int)
            @variable(m, y >= 0, start = 1)

            @objective(m, Min, -3x - y)

            @constraint(m, 3x + 2y + 10 <= 20)
            @constraint(m, x >= 1)
            @NLconstraint(m, x^2 <= 5)
            @NLconstraint(m, exp(y) + x <= 7)

            status = solve(m)

            @fact status --> :Optimal
            @fact getvalue(x) --> 2.0
        end
    end

    facts("Optimal solution with nonlinear objective test") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

            @variable(m, x >= 0, start = 1, Int)
            @variable(m, y >= 0, start = 1)

            @objective(m, Min, -3x - y)

            @constraint(m, 3x + 2y + 10 <= 20)
            @constraint(m, x >= 1)
            @NLconstraint(m, x^2 <= 5)
            @NLconstraint(m, exp(y) + x <= 7)

            status = solve(m)

            @fact status --> :Optimal
            @fact getvalue(x) --> 2.0
        end
    end

    facts("No integer variables test") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

            @variable(m, x >= 0, start = 1)
            @variable(m, y >= 0, start = 1)

            @objective(m, Min, -3x - y)

            @constraint(m, 3x + 2y + 10 <= 20)
            @constraint(m, x >= 1)

            @NLconstraint(m, x^2 <= 5)
            @NLconstraint(m, exp(y) + x <= 7)

            @fact_throws ErrorException solve(m)
        end
    end

    facts("Maximization problem") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

            @variable(m, x >= 0, start = 1, Int)
            @variable(m, y >= 0, start = 1)

            @objective(m, Max, 3x + y)

            @constraint(m, 3x + 2y + 10 <= 20)
            @NLconstraint(m, x^2 <= 9)

            status = solve(m)

            @fact round(getobjectivevalue(m) - 9.5) --> 0.0
        end
    end

    facts("Maximization problem with nonlinear function") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=0))

            @variable(m, x >= 0, start = 1, Int)
            @variable(m, y >= 0, start = 1)

            @objective(m, Max, -x^2 - y)

            @constraint(m, x + 2y >= 4)
            @NLconstraint(m, x^2 <= 9)

            status = solve(m)

            @fact round(getobjectivevalue(m) + 2.0) --> 0.0
        end
    end
end
