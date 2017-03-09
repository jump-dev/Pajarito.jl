#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runnlp(mip_solver_drives, mip_solver, nlp_solver, log)
    @testset "Sparse matrix bug test" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=3))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @NLconstraint(m, y^2 <= 10)

        @test solve(m) == :Optimal
    end

    @testset "Convex constraint with LB and UB test" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        @test_throws ErrorException solve(m)
    end

    @testset "Infeasible NLP problem" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 >= 9)
        @NLconstraint(m, exp(y) + x <= 2)

        status = solve(m, suppress_warnings=true)

        @test status == :Infeasible
    end

    @testset "Infeasible MIP problem" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, 6x + 5y >= 30)
        @NLconstraint(m, x^2 >= 8)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve(m, suppress_warnings=true)

        @test status == :Infeasible
    end

    @testset "Solver test" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=3))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)
        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve(m)

        @test status == :Optimal
        @test isapprox(getvalue(x), 2.0)
    end

    @testset "Optimal solution with nonlinear objective test" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)
        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve(m)

        @test status == :Optimal
        @test isapprox(getvalue(x), 2.0)
    end

    @testset "No integer variables test" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)

        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        @test_throws ErrorException solve(m)
    end

    @testset "Maximization problem" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, 3x + y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 <= 9)

        status = solve(m)

        @test isapprox(getobjectivevalue(m), 9.5, atol=TOL)
    end

    @testset "Maximization problem with nonlinear function" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, -x^2 - y)

        @constraint(m, x + 2y >= 4)
        @NLconstraint(m, x^2 <= 9)

        status = solve(m)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -2.0, atol=TOL)
        @test isapprox(getobjbound(m), -2.0, atol=TOL)
    end

    @testset "Maximization problem with nonlinear function (LP/QP interface)" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log))

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, -x^2 - y)

        @constraint(m, x + 2y >= 4)
        @constraint(m, x^2 <= 9)

        status = solve(m)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -2.0, atol=1e-5)
        @test isapprox(getobjbound(m), -2.0, atol=1e-5)
    end
end
