#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Take a JuMP model and solver and solve, redirecting output
function solve_jump(testname, m, redirect)
    flush(stdout)
    flush(stderr)
    @printf "%-30s... " testname
    start_time = time()

    if redirect
        mktemp() do path,io
            out = stdout
            err = stderr
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
    flush(stdout)
    flush(stderr)

    rt_time = time() - start_time
    if isa(status, ErrorException)
        @printf ":%-16s %5.2f s\n" "ErrorException" rt_time
    else
        @printf ":%-16s %5.2f s\n" status rt_time
    end

    flush(stdout)
    flush(stderr)

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
