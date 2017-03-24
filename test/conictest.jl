#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.


# Take a solver and a CBF file basename and solve the problem and return important solve information
function solve_cbf(testname, probname, solver, redirect)
    flush(STDOUT)
    flush(STDERR)
    @printf "%-30s... " testname
    tic()

    dat = ConicBenchmarkUtilities.readcbfdata("cbf/$(probname).cbf")
    (c, A, b, con_cones, var_cones, vartypes, sense, objoffset) = ConicBenchmarkUtilities.cbftompb(dat)
    flush(STDOUT)
    flush(STDERR)

    m = MathProgBase.ConicModel(solver)

    if redirect
        mktemp() do path,io
            out = STDOUT
            err = STDERR
            redirect_stdout(io)
            redirect_stderr(io)

            MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
            MathProgBase.setvartype!(m, vartypes)
            MathProgBase.optimize!(m)

            flush(io)
            redirect_stdout(out)
            redirect_stderr(err)
        end
    else
        MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
        MathProgBase.setvartype!(m, vartypes)
        MathProgBase.optimize!(m)
    end
    flush(STDOUT)
    flush(STDERR)

    status = MathProgBase.status(m)
    time = MathProgBase.getsolvetime(m)
    objval = MathProgBase.getobjval(m)
    objbound = MathProgBase.getobjbound(m)
    sol = MathProgBase.getsolution(m)
    @printf ":%-12s %5.2f s\n" status toq()
    flush(STDOUT)
    flush(STDERR)

    return (status, time, objval, objbound, sol)
end


# SOC problems for NLP and conic algorithms
function runsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

    testname = "SOC Optimal"
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

    testname = "Optimal SOCRot"
    probname = "socrot_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -9, atol=TOL)
        @test isapprox(objbound, -9, atol=TOL)
        @test isapprox(sol, [1.5, 3, 3, 3], atol=TOL)
    end

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
    testname = "Timeout 1st MIP"
    probname = "tls5"
    @testset "$testname" begin
        solver_timeout = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            timeout=15.)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_timeout, redirect)

        @test time < 60.
        @test status == :UserLimit
    end
    
    testname = "SOC dualize"
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

    testname = "Infeas none"
    probname = "soc_infeas_bin"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=false, soc_abslift=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end
end

# Exp+SOC problems for NLP and conic algorithms
function runexpsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

    testname = "ExpSOC optimal"
    probname = "expsoc_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -8, atol=TOL)
        @test isapprox(objbound, -8, atol=TOL)
        @test isapprox(sol[1:2], [2, 2], atol=TOL)
    end

    testname = "ExpSOC larger"
    probname = "expsoc_larger"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.609438, atol=TOL)
        @test isapprox(objbound, -7.609438, atol=TOL)
        @test isapprox(sol[1:2], [2, 1.609438], atol=TOL)
    end

    testname = "No disagg, no L1"
    probname = "expsoc_larger"
    @testset "$testname" begin
        solver_nodisagg = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, soc_disagg=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_nodisagg, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.609438, atol=TOL)
        @test isapprox(objbound, -7.609438, atol=TOL)
        @test isapprox(sol[1:2], [2, 1.609438], atol=TOL)
    end

    testname = "Cone composition"
    probname = "expsoc_compose"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end
end

# Exp+SOC problems for conic algorithm
function runexpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "ExpSOC no init cuts"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "No init cuts, no disagg"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_exp=false, init_soc_one=false, init_soc_inf=false, soc_disagg=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "No init cuts, no disagg, abs"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_exp=false, init_soc_one=false, init_soc_inf=false, soc_disagg=false, soc_abslift=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "Viol cuts only"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            viol_cuts_only=true, init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "No scaling"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            scale_subp_cuts=false, init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "Primal cuts always"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            prim_cuts_always=true, init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "Primal cuts only"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            prim_cuts_only=true, init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end

    testname = "No conic solver"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, log_level=log_level,
            solve_relax=false, solve_subp=false, prim_cuts_only=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end
end

# SDP+SOC problems for conic algorithm
function runsdpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

    testname = "SDPSOC optimal"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Suboptimal MIP solves"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver_subopt = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            mip_subopt_count=3, mip_subopt_solver=mip_solver)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_subopt, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Unbounded"
    probname = "sdpsoc_unbounded"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Unbounded
    end

    testname = "Infeasible"
    probname = "sdpsoc_infeasible"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "No init cuts"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, init_soc_inf=false, init_sdp_lin=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "No eig cuts"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, init_soc_inf=false, init_sdp_lin=false, sdp_eig=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Dualize"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            dualize_relax=true, dualize_subp=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Viol cuts only"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, init_soc_inf=false, init_sdp_lin=false, viol_cuts_only=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "No scaling"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_soc_one=false, init_soc_inf=false, init_sdp_lin=false, scale_subp_cuts=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Primal cuts assist"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            prim_cuts_assist=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "Primal cuts only"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            prim_cuts_only=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "No conic solver"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, log_level=log_level,
            solve_relax=false, solve_subp=false, prim_cuts_only=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

    testname = "SDP integer Aopt"
    probname = "sdp_Aopt"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 8.955043, atol=TOL)
        @test isapprox(objbound, 8.955043, atol=TOL)
        @test isapprox(sol[1:8], [0, 3, 2, 2, 0, 3, 0, 2], atol=TOL)
    end

    testname = "SDP integer Eopt"
    probname = "sdp_Eopt"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -0.2342348, atol=TOL)
        @test isapprox(objbound, -0.2342348, atol=TOL)
        @test isapprox(sol[1:8], [0, 3, 2, 3, 0, 3, 0, 1], atol=TOL)
    end
end

# SDP+Exp problems for conic algorithm
function runsdpexpconic(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "ExpSDP integer Dopt"
    probname = "expsdp_Dopt"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 1.868872, atol=TOL)
        @test isapprox(objbound, 1.868872, atol=TOL)
        @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
    end

    testname = "No init cuts Dopt"
    probname = "expsdp_Dopt"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_exp=false, init_sdp_lin=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 1.868872, atol=TOL)
        @test isapprox(objbound, 1.868872, atol=TOL)
        @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
    end

    testname = "Primal cuts only Dopt"
    probname = "expsdp_Dopt"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            init_exp=false, init_sdp_lin=false, prim_cuts_only=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 1.868872, atol=TOL)
        @test isapprox(objbound, 1.868872, atol=TOL)
        @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
    end
end

# Exp+SOC problems for conic algorithm with MISOCP
function runexpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
        soc_in_mip=true)

    testname = "ExpSOC SOC in MIP"
    probname = "expsoc_optimal"
    @testset "$testname" begin

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -8, atol=TOL)
        @test isapprox(objbound, -8, atol=TOL)
        @test isapprox(sol[1:2], [2, 2], atol=TOL)
    end

    testname = "SOC in MIP, suboptimal MIP"
    probname = "socrot_optimal"
    @testset "$testname" begin
        solver_subopt = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            soc_in_mip=true, mip_subopt_count=3, mip_subopt_solver=mip_solver)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_subopt, redirect)

        @test isapprox(objval, -9, atol=TOL)
        @test isapprox(objbound, -9, atol=TOL)
        @test isapprox(sol, [1.5, 3, 3, 3], atol=TOL)
    end

    testname = "SOC in MIP, primal only"
    probname = "expsoc_larger"
    @testset "$testname" begin
        solver_primal = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            prim_cuts_only=true, init_exp=false, init_soc_one=false, init_soc_inf=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.609438, atol=TOL)
        @test isapprox(objbound, -7.609438, atol=TOL)
        @test isapprox(sol[1:2], [2, 1.609438], atol=TOL)
    end

    testname = "SOC in MIP, no conic"
    probname = "expsoc_compose"
    @testset "$testname" begin
        solver_noconic = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, log_level=log_level,
            prim_cuts_only=true, solve_relax=false, solve_subp=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -18, atol=TOL)
        @test isapprox(objbound, -18, atol=TOL)
        @test isapprox(sol[2:4], [6, -18, 0], atol=TOL)
    end
end

# SDP+SOC problems for conic algorithm with MISOCP
function runsdpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
        soc_in_mip=true)

    testname = "SDPSOC SOC in MIP"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "SOC in MIP unbounded"
    probname = "sdpsoc_unbounded"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Unbounded
    end

    testname = "SOC in MIP infeasible"
    probname = "sdpsoc_infeasible"
    @testset "$testname" begin
        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    testname = "SOC in MIP, no eig cuts"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver_noeig = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            soc_in_mip=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_noeig, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "SOC in MIP, dualize"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver_noeig = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            soc_in_mip=true, dualize_subp=true, dualize_relax=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_noeig, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "SOC in MIP, no conic solver"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver_noconic = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, log_level=log_level,
            prim_cuts_only=true, solve_relax=false, solve_subp=false)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver_noconic, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "SDP init SOC cuts optimal"
    probname = "sdpsoc_optimal"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            sdp_eig=true, init_sdp_soc=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, -7.5, atol=TOL)
        @test isapprox(objbound, -7.5, atol=TOL)
        @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
    end

    testname = "SDP init SOC cuts infeasible"
    probname = "sdpsoc_infeasible"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            sdp_eig=true, init_sdp_soc=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Infeasible
    end

    # Only run SOC cut tests if iterative algorithm, because cannot add SOC cuts during MSD
    if !mip_solver_drives
        testname = "SDP SOC eig cuts optimal"
        probname = "sdpsoc_optimal"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, -7.5, atol=TOL)
            @test isapprox(objbound, -7.5, atol=TOL)
            @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
        end

        testname = "SDP SOC eig cuts infeasible"
        probname = "sdpsoc_infeasible"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Infeasible
        end

        testname = "SDP SOC cuts, no conic optimal"
        probname = "sdpsoc_optimal"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true, prim_cuts_only=true, solve_relax=false, solve_subp=false)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, -7.5, atol=TOL)
            @test isapprox(objbound, -7.5, atol=TOL)
            @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
        end

        testname = "SDP SOC full cuts optimal"
        probname = "sdpsoc_optimal"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, -7.5, atol=TOL)
            @test isapprox(objbound, -7.5, atol=TOL)
            @test isapprox(sol[1:6], [2, 0.5, 1, 1, 2, 2], atol=TOL)
        end

        testname = "SDP SOC full cuts infeasible"
        probname = "sdpsoc_infeasible"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Infeasible
        end

        testname = "SDP SOC eig cuts Aopt"
        probname = "sdp_Aopt"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, 8.955043, atol=TOL)
            @test isapprox(objbound, 8.955043, atol=TOL)
            @test isapprox(sol[1:8], [0, 3, 2, 2, 0, 3, 0, 2], atol=TOL)
        end

        testname = "SDP SOC eig cuts Eopt"
        probname = "sdp_Eopt"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, -0.2342348, atol=TOL)
            @test isapprox(objbound, -0.2342348, atol=TOL)
            @test isapprox(sol[1:8], [0, 3, 2, 3, 0, 3, 0, 1], atol=TOL)
        end
    end
end

# SDP+Exp problems for conic algorithm with MISOCP
function runsdpexpconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    testname = "ExpSDP init SOC cuts Dopt"
    probname = "expsdp_Dopt"
    @testset "$testname" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
            sdp_eig=true, sdp_soc=false, init_sdp_soc=true)

        (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

        @test status == :Optimal
        @test isapprox(objval, 1.868872, atol=TOL)
        @test isapprox(objbound, 1.868872, atol=TOL)
        @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
    end

    # Only run SOC cut tests if iterative algorithm, because cannot add SOC cuts during MSD
    if !mip_solver_drives
        testname = "SDP SOC eig cuts Dopt"
        probname = "expsdp_Dopt"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=true, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, 1.868872, atol=TOL)
            @test isapprox(objbound, 1.868872, atol=TOL)
            @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
        end

        testname = "SDP SOC full cuts Dopt"
        probname = "expsdp_Dopt"
        @testset "$testname" begin
            solver = PajaritoSolver(mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level,
                sdp_eig=false, sdp_soc=true)

            (status, time, objval, objbound, sol) = solve_cbf(testname, probname, solver, redirect)

            @test status == :Optimal
            @test isapprox(objval, 1.868872, atol=TOL)
            @test isapprox(objbound, 1.868872, atol=TOL)
            @test isapprox(sol[end-7:end], [0, 3, 3, 2, 0, 3, 0, 1], atol=TOL)
        end
    end
end
