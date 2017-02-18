#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# SOC problems for NLP and conic algorithms
function runsocboth(mip_solver_drives, mip_solver, cont_solver, log)
    @testset "Maximize" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x <= 10,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        @test isapprox(problem.optval, 9.0, atol=TOL)
        @test problem.status == :Optimal
    end

    @testset "Infeasible" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x >= 4,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "Equality constraint" begin
        # max  y + z
        # st   x == 1
        #     (x,y,z) in SOC
        #      x in {0,1}
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        MathProgBase.loadproblem!(m,
        [ 0.0, -1.0, -1.0],
        [ 1.0  0.0  0.0;
         -1.0  0.0  0.0;
          0.0 -1.0  0.0;
          0.0  0.0 -1.0],
        [ 1.0, 0.0, 0.0, 0.0],
        Any[(:Zero,1:1),(:SOC,2:4)],
        Any[(:Free,[1,2,3])])
        MathProgBase.setvartype!(m, [:Int,:Cont,:Cont])

        MathProgBase.optimize!(m)

        @test MathProgBase.status(m) == :Optimal
        @test isapprox(MathProgBase.getobjval(m), -sqrt(2.0), atol=TOL)
        @test isapprox(MathProgBase.getobjbound(m), -sqrt(2.0), atol=TOL)

        vals = MathProgBase.getsolution(m)
        @test isapprox(vals[1], 1, atol=TOL)
        @test isapprox(vals[2], 1.0/sqrt(2.0), atol=TOL)
        @test isapprox(vals[3], 1.0/sqrt(2.0), atol=TOL)
    end

    @testset "Zero cones" begin
        # Same as "Variable not in zero cone problem" but with variables 2 and 4 added and in zero cones
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        MathProgBase.loadproblem!(m,
        [ 0.0, 0.0, -1.0, 1.0, -1.0],
        [ 1.0  1.0  0.0  0.0  0.0;
         -1.0  0.0  0.0 -0.5  0.0;
          0.0  2.0 -1.0  0.0  0.0;
          0.0  0.0  0.0 0.5  -1.0],
        [ 1.0, 0.0, 0.0, 0.0],
        Any[(:Zero,1:1),(:SOC,2:4)],
        Any[(:Free,[1,3,5]),(:Zero,[2,4])])
        MathProgBase.setvartype!(m, [:Int,:Int,:Cont,:Cont,:Cont])

        MathProgBase.optimize!(m)

        @test MathProgBase.status(m) == :Optimal
        @test isapprox(MathProgBase.getobjval(m), -sqrt(2.0), atol=TOL)

        vals = MathProgBase.getsolution(m)
        @test isapprox(vals[1], 1, atol=TOL)
        @test isapprox(vals[2], 0, atol=TOL)
        @test isapprox(vals[3], 1.0/sqrt(2.0), atol=TOL)
        @test isapprox(vals[4], 0.0, atol=TOL)
        @test isapprox(vals[5], 1.0/sqrt(2.0), atol=TOL)
    end

    @testset "Rotated SOC" begin
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        c = [-3.0, 0.0, 0.0, 0.0]
        A = zeros(4,4)
        A[1,1] = 1.0
        A[2,2] = 1.0
        A[3,3] = 1.0
        A[4,1] = 1.0
        A[4,4] = -1.0
        b = [10.0, 1.5, 3.0, 0.0]
        constr_cones = Any[(:NonNeg,[1,2,3]),(:Zero,[4])]
        var_cones = Any[(:SOCRotated,[2,3,1]),(:Free,[4])]
        vartypes = [:Cont, :Cont, :Cont, :Int]
        MathProgBase.loadproblem!(m, c, A, b, constr_cones, var_cones)
        MathProgBase.setvartype!(m, vartypes)

        MathProgBase.optimize!(m)

        @test isapprox(MathProgBase.getobjval(m), -9.0, atol=TOL)
        @test isapprox(MathProgBase.getobjbound(m), -9.0, atol=TOL)
    end

    @testset "No continuous variables, infeasible" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        dim = 5

        @variable(m, x[1:dim], Bin)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= sqrt(dim-1)/2)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
    end
end

# SOC problems for conic algorithm
function runsocconic(mip_solver_drives, mip_solver, cont_solver, log)
    @testset "Infinite duality gap failure" begin
        # Example of polyhedral OA failure due to infinite duality gap from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
        # min  z
        # st   x == 0
        #     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
        #      x in {0,1}

        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=true, mip_solver=CplexSolver(), cont_solver=ECOSSolver(), log_level=2))

        MathProgBase.loadproblem!(m,
        [ 0.0, 0.0, 1.0],
        [ -1.0  0.0  0.0;
        -1.0  0.0  0.0;
        0.0 -1.0  0.0;
        0.0  0.0 -1.0],
        [ 0.0, 0.0, 0.0, 0.0],
        Any[(:Zero,1:1),(:SOCRotated,2:4)],
        Any[(:Free,[1,2,3])])
        MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])

        MathProgBase.optimize!(m)

        @test MathProgBase.status(m) == :ConicFailure
    end

    @testset "Hijazi: no init soc" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=false, init_soc_inf=false))

        dim = 3

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
    end

    @testset "Hijazi: L1, disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=true, soc_disagg=true, soc_abslift=false))

        dim = 5

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 1
    end

    @testset "Hijazi: L1, disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=true, soc_disagg=true, soc_abslift=true))

        dim = 5

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 1
    end

    @testset "Hijazi: L1, no disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=true, soc_disagg=false, soc_abslift=true))

        dim = 5

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 1
    end

    @testset "Hijazi: no L1, no disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=false, soc_disagg=false, soc_abslift=false))

        dim = 3

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 2^dim + 1
    end

    @testset "Hijazi: no L1, disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=false, soc_disagg=true, soc_abslift=false))

        dim = 4

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 3
    end

    @testset "Hijazi: no L1, disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=false, soc_disagg=true, soc_abslift=true))

        dim = 4

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 1
    end

    @testset "Hijazi: no L1, no disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_soc_one=false, soc_disagg=false, soc_abslift=true))

        dim = 4

        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        status = solve(m)
        @test status == :Infeasible
        internalmodel(m).logs[:n_mip] = 1
    end
end

# Exp+SOC problems for NLP and conic algorithms
function runexpsocboth(mip_solver_drives, mip_solver, cont_solver, log)
    @testset "Exp and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "No integer variables" begin
        x = Convex.Variable(1)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       @test_throws ErrorException Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))
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

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 2.0, atol=TOL)
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

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, soc_disagg=false, init_soc_one=false, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 2.0, atol=TOL)
    end

    @testset "Cone composition" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end
end

# Exp+SOC problems for conic algorithm
function runexpsocconic(mip_solver_drives, mip_solver, cont_solver, log)
    @testset "No init exp" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_exp=false))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "No init exp or soc" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_exp=false, init_soc_one=false, init_soc_inf=false))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "No disagg, no abslift" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, soc_disagg=false, soc_abslift=false))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "No disagg, abslift" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, soc_disagg=false, soc_abslift=true))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "Disagg, no abslift" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, soc_disagg=true, soc_abslift=false))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "Disagg, abslift" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, soc_disagg=true, soc_abslift=true))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "Dualize" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, dualize_sub=true, dualize_relax=true))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "Viol cuts only true" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, viol_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "Viol cuts only false" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, viol_cuts_only=false))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "No scaling" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, scale_subp_cuts=false))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "No primal cuts assist" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_assist=false))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "Primal cuts always" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_always=true))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    @testset "Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end
end

# SDP+SOC problems for conic algorithm
function runsdpsocconic(mip_solver_drives, mip_solver, cont_solver, log)
    @testset "SDP and SOC" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "No init sdp" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, init_sdp_lin=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "No eig cuts" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, sdp_eig=false))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "Dualize" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, dualize_sub=true, dualize_relax=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "Viol cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, viol_cuts_only=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "No scaling" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, scale_subp_cuts=false))

        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "No primal cuts assist" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_assist=false))

        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "Primal cuts always" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_always=true))

        @test problem.status == :Optimal
        @test isapprox(problem.optval, 8.0, atol=TOL)
    end

    @testset "Primal cuts only" begin
        x = Convex.Variable(1, :Int)
        y = Convex.Variable(1, Convex.Positive())
        z = Convex.Semidefinite(2)

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            x^2 <= 4,
                            y >= z[2,2])

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log, prim_cuts_assist=true, prim_cuts_always=true, prim_cuts_only=true))

        @test isapprox(problem.optval, 8.0, atol=TOL)
    end
end
