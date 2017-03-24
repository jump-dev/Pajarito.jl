#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runnlp(mip_solver_drives, mip_solver, nlp_solver, log_level, redirect)
    if redirect
        log_level = 0
    end
    solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=nlp_solver, log_level=log_level))

    @testset "Optimal" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @NLconstraint(m, y^2 <= 10)

        @test solve(m, suppress_warnings=true) == :Optimal
    end

    @testset "Nonconvex error" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        @test_throws ErrorException solve(m, suppress_warnings=true)
    end

    @testset "Infeasible 1" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 >= 9)
        @NLconstraint(m, exp(y) + x <= 2)

        status = solve(m, suppress_warnings=true)

        @test status == :Infeasible
    end

    @testset "Infeasible 2" begin
        m = Model(solver)

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

    @testset "Optimal 2" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)
        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getvalue(x), 2.0)
    end

    @testset "Continuous error" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)

        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        @test_throws ErrorException solve(m, suppress_warnings=true)
    end

    @testset "Maximization" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, 3x + y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 <= 9)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), 9.5, atol=TOL)
    end

    @testset "Nonlinear objective" begin
        m = Model(solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, -x^2 - y)

        @constraint(m, x + 2y >= 4)
        @NLconstraint(m, x^2 <= 9)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -2.0, atol=TOL)
        @test isapprox(getobjbound(m), -2.0, atol=TOL)
    end
end
