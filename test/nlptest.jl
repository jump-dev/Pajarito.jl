#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runnonlineartests(branch_cut, mip_solver, nlp_solver)
    algorithm = branch_cut ? "BC" : "OA"

    facts("\n\n\n\nSparse matrix bug test\n\n") do
        m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @NLconstraint(m, y^2 <= 10)

        @fact solve(m) --> :Optimal
    end

    facts("\n\n\n\nConvex constraint with LB and UB test\n\n") do
        m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        @fact_throws ErrorException solve(m)
    end

    facts("\n\n\n\nInfeasible NLP problem\n\n") do
        m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 >= 9)
        @NLconstraint(m, exp(y) + x <= 2)

        status = solve(m)

        @fact status --> :Infeasible
    end

    facts("\n\n\n\nInfeasible MIP problem\n\n") do
        m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

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

    facts("\n\n\n\nSolver test\n\n") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

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

    facts("\n\n\n\nOptimal solution with nonlinear objective test\n\n") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

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

    facts("\n\n\n\nNo integer variables test\n\n") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

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

    facts("\n\n\n\nMaximization problem\n\n") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

            @variable(m, x >= 0, start = 1, Int)
            @variable(m, y >= 0, start = 1)

            @objective(m, Max, 3x + y)

            @constraint(m, 3x + 2y + 10 <= 20)
            @NLconstraint(m, x^2 <= 9)

            status = solve(m)

            @fact round(getobjectivevalue(m) - 9.5) --> 0.0
        end
    end

    facts("\n\n\n\nMaximization problem with nonlinear function\n\n") do
        context("With $algorithm, $(typeof(mip_solver)) and $(typeof(nlp_solver))") do
            m = Model(solver=PajaritoSolver(branch_cut=branch_cut, solver_mip=mip_solver, solver_cont=nlp_solver))

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
