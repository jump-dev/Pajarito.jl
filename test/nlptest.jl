#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Take a JuMP model and solver and solve, redirecting output
function solve_jump(testname, m, redirect)
    flush(STDOUT)
    flush(STDERR)
    @printf "%-30s... " testname
    tic()

    if redirect
        mktemp() do path,io
            out = STDOUT
            err = STDERR
            redirect_stdout(io)
            redirect_stderr(io)

            status = try
                solve(m)
            catch e
                e
            end

            flush(io)
            redirect_stdout(out)
            redirect_stderr(err)
        end
    else
        status = try
            solve(m)
        catch e
            e
        end
    end
    flush(STDOUT)
    flush(STDERR)

    if isa(status, ErrorException)
        @printf ":%-16s %5.2f s\n" "ErrorException" toq()
    else
        @printf ":%-16s %5.2f s\n" status toq()
    end

    flush(STDOUT)
    flush(STDERR)

    return status
end


# Quadratically constrained problems compatible with MathProgBase ConicToLPQPBridge
function run_qp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver=PajaritoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=(redirect ? 0 : 3))

    testname = "QP optimal"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, Int)
        @variable(m, y >= 0)
        @variable(m, 0 <= u <= 10, Int)
        @variable(m, w == 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @constraint(m, y^2 <= u*w)

        status = solve_jump(testname, m, redirect)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -12.162277, atol=TOL)
        @test isapprox(getobjbound(m), -12.162277, atol=TOL)
        @test isapprox(getvalue(x), 3, atol=TOL)
        @test isapprox(getvalue(y), 3.162277, atol=TOL)
    end

    testname = "QP maximize"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, Int)
        @variable(m, y >= 0)
        @variable(m, 0 <= u <= 10, Int)
        @variable(m, w == 1)

        @objective(m, Max, 3x + y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x^2 <= u*w)

        status = solve_jump(testname, m, redirect)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), 9.5, atol=TOL)
        @test isapprox(getobjbound(m), 9.5, atol=TOL)
        @test isapprox(getvalue(x), 3, atol=TOL)
        @test isapprox(getvalue(y), 0.5, atol=TOL)
    end
end

# NLP problems NOT compatible with MathProgBase ConicToLPQPBridge
function run_nlp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver=PajaritoSolver(timeout=120., mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=(redirect ? 0 : 3))

    testname = "Nonconvex error"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        status = solve_jump(testname, m, redirect)

        @test isa(status, ErrorException)
    end

    testname = "Optimal"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)
        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test status == :Optimal
        @test isapprox(getvalue(x), 2.0)
    end

    testname = "Infeasible 1"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 >= 9)
        @NLconstraint(m, exp(y) + x <= 2)

        status = solve_jump(testname, m, redirect)

        @test status == :Infeasible
    end

    testname = "Infeasible 2"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, 6x + 5y >= 30)
        @NLconstraint(m, x^2 >= 8)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test status == :Infeasible
    end

    testname = "Continuous error"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x >= 1)

        @NLconstraint(m, x^2 <= 5)
        @NLconstraint(m, exp(y) + x <= 7)

        status = solve_jump(testname, m, redirect)

        @test isa(status, ErrorException)
    end

    testname = "Maximization"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, 3x + y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, x^2 <= 9)

        status = solve_jump(testname, m, redirect)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), 9.5, atol=TOL)
    end

    testname = "Nonlinear objective"
    @testset "$testname" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Max, -x^2 - y)

        @constraint(m, x + 2y >= 4)
        @NLconstraint(m, x^2 <= 9)

        status = solve_jump(testname, m, redirect)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -2.0, atol=TOL)
        @test isapprox(getobjbound(m), -2.0, atol=TOL)
    end
end
