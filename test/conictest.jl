#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.


# Take a solver and a CBF file basename and solve the problem and return important solve information
function solve_cbf(testname, probname, solver, redirect)
    @printf "%-30s... " testname
    tic()

    dat = ConicBenchmarkUtilities.readcbfdata("cbf/$(probname).cbf")
    (c, A, b, con_cones, var_cones, vartypes, sense, objoffset) = ConicBenchmarkUtilities.cbftompb(dat)

    m = MathProgBase.ConicModel(solver)

    if redirect
        mktemp() do path,io
            TT = STDOUT
            redirect_stdout(io)

            MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
            MathProgBase.setvartype!(m, vartypes)
            MathProgBase.optimize!(m)

            redirect_stdout(TT)
        end
    else
        MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
        MathProgBase.setvartype!(m, vartypes)
        MathProgBase.optimize!(m)
    end

    status = MathProgBase.status(m)
    time = MathProgBase.getsolvetime(m)
    objval = MathProgBase.getobjval(m)
    objbound = MathProgBase.getobjbound(m)
    sol = MathProgBase.getsolution(m)

    @printf ":%-12s %5.2f s\n" status toq()

    return (status, time, objval, objbound, sol)
end


# SOC problems for NLP and conic algorithms
function runsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver,
        log_level=3)

    testname = "Optimal"
    probname = "soc_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(sol[1], 3, atol=TOL)
        @test isapprox(objval, -9, atol=TOL)
    end

    testname = "Infeasible"
    probname = "soc_infeasible"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Unbounded"
    probname = "soc_unbounded"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Unbounded
    end

    testname = "Timeout 1st MIP"
    probname = "tls5"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver,
            log_level=3, timeout=15.)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test time < 60.
        @test status == :UserLimit
    end

    testname = "Optimal SOCRot"
    probname = "socrot_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -9, atol=TOL)
        @test isapprox(objbound, -9, atol=TOL)
        @test isapprox(sol, [1.5, 3, 3, 3], atol=TOL)
    end

    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

    testname = "Infeasible SOCRot"
    probname = "socrot_infeasible"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Equality constraint"
    probname = "soc_equality"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -sqrt(2), atol=TOL)
        @test isapprox(objbound, -sqrt(2), atol=TOL)
        @test isapprox(sol, [1, 1/sqrt(2), 1/sqrt(2)], atol=TOL)
    end

    testname = "Zero cones"
    probname = "soc_zero"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -sqrt(2), atol=TOL)
        @test isapprox(objbound, -sqrt(2), atol=TOL)
        @test isapprox(sol, [1, 1/sqrt(2), 1/sqrt(2), 0, 0], atol=TOL)
    end

    testname = "Infeasible all binary"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end
end

# SOC problems for conic algorithm
function runsocconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "Dualize SOC"
    probname = "soc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            dualize_subp=true, dualize_relax=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(sol[1], 3, atol=TOL)
        @test isapprox(objval, -9, atol=TOL)
    end

    testname = "Suboptimal MIP solves"
    probname = "soc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            mip_subopt_count=3, mip_subopt_solver=mip_solver)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(sol[1], 3, atol=TOL)
        @test isapprox(objval, -9, atol=TOL)
    end

    testname = "Dualize SOCRot"
    probname = "socrot_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            dualize_subp=true, dualize_relax=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -9, atol=TOL)
        @test isapprox(objbound, -9, atol=TOL)
        @test isapprox(sol, [1.5, 3, 3, 3], atol=TOL)
    end

    testname = "Infeas L1, disagg"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=true, soc_disagg=true, soc_abslift=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas L1, disagg, abs"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=true, soc_disagg=true, soc_abslift=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas L1, abs"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=true, soc_disagg=false, soc_abslift=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas none"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=false, soc_abslift=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas disagg"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=true, soc_abslift=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas disagg, abs"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=true, soc_abslift=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "Infeas abs"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=false, soc_abslift=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end
end

# Exp+SOC problems for NLP and conic algorithms
function runexpsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "Exp and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test isapprox(x.value, 2.0, atol=TOL)
       @test isapprox(y.value, 2.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "No integer variables" begin
        x = Convex.Variable(1)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       @test_throws ErrorException Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))
    end

    @testset "More constraints" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1)

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           y >= 0,
                           3x + 2y <= 10,
                           x^2 <= 5,
                           exp(y) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        @test problem.status == :Optimal
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 1.609438, atol=TOL)
    end

    @testset "No SOC disaggregation" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1)

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           y >= 0,
                           3x + 2y <= 10,
                           x^2 <= 5,
                           exp(y) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, soc_disagg=false, init_soc_one=false, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        @test problem.status == :Optimal
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 1.609438, atol=TOL)
    end

    @testset "Cone composition" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end
end

# Exp+SOC problems for conic algorithm
function runexpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "No init exp" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3, init_exp=false))

       @test problem.status == :Optimal
       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test isapprox(x.value, 2.0, atol=TOL)
       @test isapprox(y.value, 2.0, atol=TOL)
    end

    @testset "No init exp or soc" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_exp=false, init_soc_one=false, init_soc_inf=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end

    @testset "No disagg, abs" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_disagg=false, soc_abslift=true))

       @test problem.status == :Optimal
       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test isapprox(x.value, 2.0, atol=TOL)
       @test isapprox(y.value, 2.0, atol=TOL)
    end

    @testset "Disagg, no abs" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_disagg=true, soc_abslift=false))

       @test problem.status == :Optimal
       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test isapprox(x.value, 2.0, atol=TOL)
       @test isapprox(y.value, 2.0, atol=TOL)
    end

    @testset "Disagg, abs" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_disagg=true, soc_abslift=true))

       @test problem.status == :Optimal
       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test isapprox(x.value, 2.0, atol=TOL)
       @test isapprox(y.value, 2.0, atol=TOL)
    end

    @testset "Viol cuts only true" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, viol_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end

    @testset "Viol cuts only false" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, viol_cuts_only=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end

    @testset "No scaling" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, scale_subp_cuts=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end

    @testset "Primal cuts always" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_always=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end

    @testset "Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end
end

# SDP+SOC problems for conic algorithm
function runsdpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "SDP and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Subopt solves" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, mip_subopt_count=3, mip_subopt_solver=mip_solver, cont_solver=cont_solver, log_level=3))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Unbounded" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(z[1,1] - x,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level), verbose=false)

        @test problem.status == :Unbounded
    end

    @testset "Infeasible" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 2,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 2,
                            y >= z[2,2] + z[1,1])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "No integer variables" begin
        x = Convex.Variable(1)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

       @test_throws ErrorException Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))
    end

    @testset "No init sdp" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_lin=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "No eig cuts" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_eig=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Dualize" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, dualize_subp=true, dualize_relax=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Viol cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, viol_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "No scaling" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, scale_subp_cuts=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "No primal cuts assist" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_assist=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Primal cuts always" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_always=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Convex.jl A-opt" begin
        # A-optimal design
        #   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
        #   subject to  lambda >= 0, 1'*lambda = 1
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)
        u = Convex.Variable(q)

        aOpt = Convex.minimize(
            sum(u),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )
        E = eye(q)
        for i in 1:q
        	aOpt.constraints += Convex.isposdef([Q E[:,i]; E[i,:]' u[i]])
        end

        Convex.solve!(aOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        @test aOpt.status == :Optimal
        @test isapprox(aOpt.optval, 8.955043, atol=TOL)
        @test isapprox(Convex.evaluate(sum(u)), aOpt.optval, atol=TOL)
        @test isapprox(np.value, [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    end

    @testset "JuMP.jl A-opt" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    end

    @testset "Convex.jl E-opt" begin
        # E-optimal design
        #   maximize    w
        #   subject to  sum_i lambda_i*vi*vi' >= w*I
        #               lambda >= 0,  1'*lambda = 1;
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)
        t = Convex.Variable()

        eOpt = Convex.maximize(
            t,
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax,
            Convex.isposdef(Q - t * eye(q))
        )

        Convex.solve!(eOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        @test eOpt.status == :Optimal
        @test isapprox(eOpt.optval, 0.2342348, atol=TOL)
        @test isapprox(Convex.evaluate(t), eOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    @testset "JuMP.jl E-opt" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 0.2342348, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    # @testset "No relax solve: A-opt" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, solve_relax=false))
    #
    #     (q, p, n, nmax) = (4, 8, 12, 3)
    #     V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
    #
    #     np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    #     @constraint(aOpt, sum(np) <= n)
    #     u = @variable(aOpt, [i=1:q], lowerbound=0)
    #     @objective(aOpt, Min, sum(u))
    #     E = eye(q)
    #     for i=1:q
    #         @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    #     end
    #
    #     @test solve(aOpt, suppress_warnings=true) == :Optimal
    #
    #     @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    # end

    @testset "No subp solve: A-opt" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, solve_subp=false))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    end

    @testset "No conic solver: A-opt" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, log_level=log_level, solve_relax=false, solve_subp=false))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    end

end

# SDP+Exp problems for conic algorithm
function runsdpexpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "Convex.jl D-opt" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)

        dOpt = Convex.maximize(
            Convex.logdet(Q),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )

        Convex.solve!(dOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=0))

        @test dOpt.status == :Optimal
        @test isapprox(dOpt.optval, -1.868872, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 3.0; 2.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end
end

# Exp+SOC problems for conic algorithm with MISOCP
function runexpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "SOC in MIP: More constraints" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1)

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           y >= 0,
                           3x + 2y <= 10,
                           x^2 <= 5,
                           exp(y) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3, soc_in_mip=true))

        @test problem.status == :Optimal
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 1.609438, atol=TOL)
    end

    @testset "SOC in MIP: Cone composition" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, -18.0, atol=TOL)
        @test isapprox(x.value, 6.0, atol=TOL)
        @test isapprox(y.value, 0.0, atol=TOL)
    end
end

# SDP+SOC problems for conic algorithm with MISOCP
function runsdpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "SOC in MIP: SDP and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3, soc_in_mip=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "SOC in MIP: Unbounded" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(z[1,1] - x,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true), verbose=false)

        @test problem.status == :Unbounded
    end

    @testset "SOC in MIP: Infeasible" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 2,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 2,
                            y >= z[2,2] + z[1,1])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "SOC in MIP: No eig cuts" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true, sdp_eig=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "SOC in MIP: Dualize" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true, dualize_subp=true, dualize_relax=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "SOC in MIP: No primal cuts assist" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true, prim_cuts_assist=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "SOC in MIP: Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Full SOC: SDP and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Full SOC: Infeasible" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 2,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 2,
                            y >= z[2,2] + z[1,1])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "Init SOC and eig SOC: SDP and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Init SOC and eig SOC: Infeasible" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 2,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 2,
                            y >= z[2,2] + z[1,1])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "Init SOC and eig SOC: Dualize" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true, dualize_subp=true, dualize_relax=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Init SOC and eig SOC: No primal cuts assist" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true, prim_cuts_assist=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    @testset "Init SOC and eig SOC: Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y - z[1,1],
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            z[1,2] >= 1,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 7.5, atol=TOL)
        @test isapprox(x.value, 2.0, atol=TOL)
        @test isapprox(y.value, 2.0, atol=TOL)
        @test isapprox(z.value, [0.5 1.0; 1.0 2.0], atol=TOL)
    end

    # @testset "SOC eig cuts: A-opt" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true))
    #
    #     (q, p, n, nmax) = (4, 8, 12, 3)
    #     V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
    #
    #     np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    #     @constraint(aOpt, sum(np) <= n)
    #     u = @variable(aOpt, [i=1:q], lowerbound=0)
    #     @objective(aOpt, Min, sum(u))
    #     E = eye(q)
    #     for i=1:q
    #         @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    #     end
    #
    #     @test solve(aOpt, suppress_warnings=true) == :Optimal
    #
    #     @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    # end

    @testset "SOC eig cuts: E-opt" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 0.2342348, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    # @testset "SOC full cuts: A-opt" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))
    #
    #     (q, p, n, nmax) = (4, 8, 12, 3)
    #     V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
    #
    #     np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    #     @constraint(aOpt, sum(np) <= n)
    #     u = @variable(aOpt, [i=1:q], lowerbound=0)
    #     @objective(aOpt, Min, sum(u))
    #     E = eye(q)
    #     for i=1:q
    #         @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    #     end
    #
    #     @test solve(aOpt) == :Optimal
    #
    #     @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    # end

    @testset "Init SOC cuts: E-opt" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 0.2342348, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    # @testset "Init SOC cuts: A-opt" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true))
    #
    #     (q, p, n, nmax) = (4, 8, 12, 3)
    #     V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
    #
    #     np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    #     @constraint(aOpt, sum(np) <= n)
    #     u = @variable(aOpt, [i=1:q], lowerbound=0)
    #     @objective(aOpt, Min, sum(u))
    #     E = eye(q)
    #     for i=1:q
    #         @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    #     end
    #
    #     @test solve(aOpt) == :Optimal
    #
    #     @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    # end

    @testset "SOC full cuts: E-opt" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 0.2342348, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    # @testset "Init and eig SOC cuts: A-opt" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))
    #
    #     (q, p, n, nmax) = (4, 8, 12, 3)
    #     V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
    #
    #     np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    #     @constraint(aOpt, sum(np) <= n)
    #     u = @variable(aOpt, [i=1:q], lowerbound=0)
    #     @objective(aOpt, Min, sum(u))
    #     E = eye(q)
    #     for i=1:q
    #         @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    #     end
    #
    #     @test solve(aOpt) == :Optimal
    #
    #     @test isapprox(getobjectivevalue(aOpt), 8.955043, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [-0.0; 3.0; 2.0; 2.0; -0.0; 3.0; -0.0; 2.0], atol=TOL)
    # end

    @testset "Init and eig SOC cuts: E-opt" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))

        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 0.2342348, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [0.0; 3.0; 2.0; 3.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end
end

# SDP+Exp problems for conic algorithm with MISOCP
function runsdpexpconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    @testset "SOC eig cuts: D-opt" begin
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)

        dOpt = Convex.maximize(
            Convex.logdet(Q),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )

        Convex.solve!(dOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3, sdp_soc=true))

        @test dOpt.status == :Optimal
        @test isapprox(dOpt.optval, -1.868872, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 3.0; 2.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    @testset "SOC full cuts: D-opt" begin
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)

        dOpt = Convex.maximize(
            Convex.logdet(Q),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )

        Convex.solve!(dOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))

        @test dOpt.status == :Optimal
        @test isapprox(dOpt.optval, -1.868872, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 3.0; 2.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    @testset "Init SOC cuts: D-opt" begin
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)

        dOpt = Convex.maximize(
            Convex.logdet(Q),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )

        Convex.solve!(dOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true))

        @test dOpt.status == :Optimal
        @test isapprox(dOpt.optval, -1.868872, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 3.0; 2.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end

    @testset "Init and eig SOC cuts: D-opt" begin
        (q, p, n, nmax) = (4, 8, 12, 3)
        V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]

        np = Convex.Variable(p, :Int)
        Q = Convex.Variable(q, q)

        dOpt = Convex.maximize(
            Convex.logdet(Q),
            Q == V * diagm(np./n) * V',
            sum(np) <= n,
            np >= 0,
            np <= nmax
        )

        Convex.solve!(dOpt, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))

        @test dOpt.status == :Optimal
        @test isapprox(dOpt.optval, -1.868872, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [0.0; 3.0; 3.0; 2.0; 0.0; 3.0; 0.0; 1.0], atol=TOL)
    end
end
