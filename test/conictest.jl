#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# SOC problems for NLP and conic algorithms
function runsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level)
    @testset "Maximize" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x <= 10,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3))

        @test isapprox(x.value, 3.0, atol=TOL)
        @test isapprox(problem.optval, 9.0, atol=TOL)
        @test problem.status == :Optimal
    end

    @testset "Infeasible" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x >= 4,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "Unbounded" begin
        x = Convex.Variable(1, :Int)
        t = Convex.Variable(1, :Int)

        problem = Convex.maximize(-3x + t,
                            x <= 10,
                            x^2 <= 2t,
                            t >= 5)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level), verbose=false)

        @test problem.status == :Unbounded
    end

    @testset "Equality constraint" begin
        # max  y + z
        # st   x == 1
        #     (x,y,z) in SOC
        #      x in {0,1}

        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

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
        @test isapprox(MathProgBase.getsolution(m), [1.0,1.0/sqrt(2.0),1.0/sqrt(2.0)], atol=TOL)
    end

    @testset "Zero cones" begin
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

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
        @test isapprox(MathProgBase.getobjbound(m), -sqrt(2.0), atol=TOL)
        @test isapprox(MathProgBase.getsolution(m), [1.0,0.0,1.0/sqrt(2.0),0.0,1.0/sqrt(2.0)], atol=TOL)
    end

    @testset "Rotated SOC" begin
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

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

        @test MathProgBase.status(m) == :Optimal
        @test isapprox(MathProgBase.getobjval(m), -9.0, atol=TOL)
        @test isapprox(MathProgBase.getobjbound(m), -9.0, atol=TOL)
        @test isapprox(MathProgBase.getsolution(m), [3.0,1.5,3.0,3.0], atol=TOL)
    end

    @testset "No continuous variables, infeasible" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        dim = 5
        @variable(m, x[1:dim], Bin)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= sqrt(dim-1)/2)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
    end
end

# SOC problems for conic algorithm
function runsocconic(mip_solver_drives, mip_solver, cont_solver, log_level)
    @testset "Dualize SOC" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x <= 10,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=3, dualize_sub=true, dualize_relax=true))

        @test isapprox(problem.optval, 9.0, atol=TOL)
        @test isapprox(x.value, 3.0, atol=TOL)
        @test problem.status == :Optimal
    end

    @testset "Suboptimal solves" begin
        x = Convex.Variable(1, :Int)

        problem = Convex.maximize(3x,
                            x <= 10,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, mip_subopt_count=3, mip_subopt_solver=mip_solver, cont_solver=cont_solver, log_level=3, dualize_sub=true, dualize_relax=true))

        @test isapprox(problem.optval, 9.0, atol=TOL)
        @test isapprox(x.value, 3.0, atol=TOL)
        @test problem.status == :Optimal
    end

    @testset "Dualize rotated SOC" begin
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, dualize_sub=true, dualize_relax=true))

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

        @test MathProgBase.status(m) == :Optimal
        @test isapprox(MathProgBase.getobjval(m), -9.0, atol=TOL)
        @test isapprox(MathProgBase.getobjbound(m), -9.0, atol=TOL)
        @test isapprox(MathProgBase.getsolution(m), [3.0,1.5,3.0,3.0], atol=TOL)
    end

    # @testset "Infinite duality gap: primal assist" begin
    #     # Example of polyhedral OA failure due to infinite duality gap from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
    #     # min  z
    #     # st   x == 0
    #     #     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
    #     #      x in {0,1}
    #
    #     m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_disagg=false, soc_abslift=false, init_soc_one=false, init_soc_inf=false))
    #
    #     MathProgBase.loadproblem!(m,
    #     [ 0.0, 0.0, 1.0],
    #     [ -1.0  0.0  0.0;
    #     -1.0  0.0  0.0;
    #     0.0 -1.0  0.0;
    #     0.0  0.0 -1.0],
    #     [ 0.0, 0.0, 0.0, 0.0],
    #     Any[(:Zero,1:1),(:SOCRotated,2:4)],
    #     Any[(:Free,[1,2,3])])
    #     MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #     MathProgBase.optimize!(m)
    #
    #     status = MathProgBase.status(m)
    #     @test status == :CutsFailure
    # end
    #
    # @testset "Infinite duality gap: no primal assist" begin
    #     m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_assist=false, soc_disagg=false, soc_abslift=false, init_soc_one=false, init_soc_inf=false))
    #
    #     MathProgBase.loadproblem!(m,
    #     [ 0.0, 0.0, 1.0],
    #     [ -1.0  0.0  0.0;
    #     -1.0  0.0  0.0;
    #     0.0 -1.0  0.0;
    #     0.0  0.0 -1.0],
    #     [ 0.0, 0.0, 0.0, 0.0],
    #     Any[(:Zero,1:1),(:SOCRotated,2:4)],
    #     Any[(:Free,[1,2,3])])
    #     MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #     MathProgBase.optimize!(m)
    #
    #     status = MathProgBase.status(m)
    #     @test status == :CutsFailure
    # end
    #
    # @testset "Finite duality gap: primal assist" begin
    #     # Example of polyhedral OA failure due to finite duality gap, modified from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
    #     # min  z
    #     # st   x == 0
    #     #     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
    #     #      z >= -10
    #     #      x in {0,1}
    #
    #     m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_disagg=false, soc_abslift=false, init_soc_one=false, init_soc_inf=false))
    #
    #     MathProgBase.loadproblem!(m,
    #     [ 0.0, 0.0, 1.0],
    #     [ -1.0  0.0  0.0;
    #      -1.0  0.0  0.0;
    #       0.0 -1.0  0.0;
    #       0.0  0.0 -1.0;
    #       0.0  0.0 -1.0],
    #     [ 0.0, 0.0, 0.0, 0.0, 10.0],
    #     Any[(:Zero,1:1),(:SOCRotated,2:4),(:NonNeg,5:5)],
    #     Any[(:Free,[1,2,3])])
    #     MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #     MathProgBase.optimize!(m)
    #
    #     status = MathProgBase.status(m)
    #     @test status == :CutsFailure
    # end
    #
    # @testset "Finite duality gap: no primal assist" begin
    #     m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, prim_cuts_assist=false, soc_disagg=false, soc_abslift=false, init_soc_one=false, init_soc_inf=false))
    #
    #     MathProgBase.loadproblem!(m,
    #     [ 0.0, 0.0, 1.0],
    #     [ -1.0  0.0  0.0;
    #      -1.0  0.0  0.0;
    #       0.0 -1.0  0.0;
    #       0.0  0.0 -1.0;
    #       0.0  0.0 -1.0],
    #     [ 0.0, 0.0, 0.0, 0.0, 10.0],
    #     Any[(:Zero,1:1),(:SOCRotated,2:4),(:NonNeg,5:5)],
    #     Any[(:Free,[1,2,3])])
    #     MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #     MathProgBase.optimize!(m)
    #
    #     status = MathProgBase.status(m)
    #     @test status == :CutsFailure
    # end

    @testset "Hijazi: L1, disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=true, soc_disagg=true, soc_abslift=false))

        dim = 5
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        @test internalmodel(m).logs[:n_inf] == 0
    end

    @testset "Hijazi: L1, disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=true, soc_disagg=true, soc_abslift=true))

        dim = 5
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        @test internalmodel(m).logs[:n_inf] == 0
    end

    @testset "Hijazi: L1, no disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=true, soc_disagg=false, soc_abslift=true))

        dim = 5
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        @test internalmodel(m).logs[:n_inf] == 0
    end

    @testset "Hijazi: no L1, no disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=false, init_soc_inf=false, soc_disagg=false, soc_abslift=false))

        dim = 4
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        # @test internalmodel(m).logs[:n_inf] == 2^dim
    end

    @testset "Hijazi: no L1, disagg, no abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=false, init_soc_inf=false, soc_disagg=true, soc_abslift=false))

        dim = 4
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        # @test internalmodel(m).logs[:n_inf] == 2
    end

    @testset "Hijazi: no L1, disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=false, init_soc_inf=false, soc_disagg=true, soc_abslift=true))

        dim = 4
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        # @test internalmodel(m).logs[:n_inf] == 1
    end

    @testset "Hijazi: no L1, no disagg, abslift" begin
        m = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_soc_one=false, soc_disagg=false, soc_abslift=true))

        dim = 4
        @variable(m, x[1:dim], Bin)
        @variable(m, t)
        @constraint(m, t == sqrt(dim-1)/2)
        @constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
        @objective(m, Min, 0)

        @test solve(m, suppress_warnings=true) == :Infeasible
        # @test internalmodel(m).logs[:n_inf] == 1
    end

    @testset "Timeout in first MIP solve" begin
        solver = PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, timeout=15.)
        dat = ConicBenchmarkUtilities.readcbfdata("cbf/classical_200_1.cbf.gz")

        c, A, b, con_cones, var_cones, vartypes, sense, objoffset = ConicBenchmarkUtilities.cbftompb(dat)
        m = MathProgBase.ConicModel(solver)
        MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
        MathProgBase.setvartype!(m, vartypes)
        MathProgBase.optimize!(m)
        @test MathProgBase.status(m) == :UserLimit
    end
end

# Exp+SOC problems for NLP and conic algorithms
function runexpsocnlpconic(mip_solver_drives, mip_solver, cont_solver, log_level)
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
function runexpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level)
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

    @testset "No disagg, abslift" begin
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

    @testset "Disagg, no abslift" begin
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

    @testset "Disagg, abslift" begin
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
function runsdpsocconic(mip_solver_drives, mip_solver, cont_solver, log_level)
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

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, dualize_sub=true, dualize_relax=true))

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

    @testset "Convex.jl A-opt design" begin
        # A-optimal design
        #   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
        #   subject to  lambda >= 0, 1'*lambda = 1
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(aOpt.optval, 0.177181, atol=TOL)
        @test isapprox(Convex.evaluate(sum(u)), aOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,1.0,2.0,2.0], atol=TOL)
    end

    @testset "JuMP.jl A-opt design" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 0.177181, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,2.0,2.0], atol=TOL)
    end

    @testset "Convex.jl E-opt design" begin
        # E-optimal design
        #   maximize    w
        #   subject to  sum_i lambda_i*vi*vi' >= w*I
        #               lambda >= 0,  1'*lambda = 1;
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(eOpt.optval, 10.466724, atol=TOL)
        @test isapprox(Convex.evaluate(t), eOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,1.0,1.0,3.0], atol=TOL)
    end

    @testset "JuMP.jl E-opt design" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 10.466724, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,1.0,3.0], atol=TOL)
    end
end

# SDP+Exp problems for conic algorithm
function runsdpexpconic(mip_solver_drives, mip_solver, cont_solver, log_level)
    @testset "Convex.jl D-opt design" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(dOpt.optval, 9.062207, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,2.0,2.0,1.0], atol=TOL)
    end

end

# Exp+SOC problems for conic algorithm with MISOCP
function runexpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level)
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
function runsdpsocconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level)
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

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, soc_in_mip=true, dualize_sub=true, dualize_relax=true))

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

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true, dualize_sub=true, dualize_relax=true))

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

    @testset "SOC eig cuts: JuMP.jl A-opt design" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 0.177181, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,2.0,2.0], atol=TOL)
    end

    @testset "SOC eig cuts: JuMP.jl E-opt design" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 10.466724, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,1.0,3.0], atol=TOL)
    end

    @testset "SOC full cuts: JuMP.jl A-opt design" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 0.177181, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,2.0,2.0], atol=TOL)
    end

    @testset "Init SOC cuts: JuMP.jl E-opt design" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, sdp_soc=true, sdp_eig=false))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 10.466724, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,1.0,3.0], atol=TOL)
    end

    @testset "Init SOC cuts: JuMP.jl A-opt design" begin
        aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(aOpt, sum(np) <= n)
        u = @variable(aOpt, [i=1:q], lowerbound=0)
        @objective(aOpt, Min, sum(u))
        E = eye(q)
        for i=1:q
            @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
        end

        @test solve(aOpt) == :Optimal

        @test isapprox(getobjectivevalue(aOpt), 0.177181, atol=TOL)
        @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,2.0,2.0], atol=TOL)
    end

    @testset "SOC full cuts: JuMP.jl E-opt design" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 10.466724, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,1.0,3.0], atol=TOL)
    end

    # @testset "Init and eig SOC cuts: JuMP.jl A-opt design" begin
    #     aOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))
    #
    #     n = 7
    #     nmax = 3
    #     V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
    #     (q, p) = size(V)
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
    #     @test isapprox(getobjectivevalue(aOpt), 0.177181, atol=TOL)
    #     @test isapprox(getvalue(sum(u)), getobjectivevalue(aOpt), atol=TOL)
    #     @test isapprox(getvalue(np), [2.0,1.0,2.0,2.0], atol=TOL)
    # end

    @testset "Init and eig SOC cuts: JuMP.jl E-opt design" begin
        eOpt = Model(solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=log_level, init_sdp_soc=true, sdp_soc=true))

        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

        np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
        @constraint(eOpt, sum(np) <= n)
        t = @variable(eOpt)
        @objective(eOpt, Max, t)
        @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

        @test solve(eOpt, suppress_warnings=true) == :Optimal

        @test isapprox(getobjectivevalue(eOpt), 10.466724, atol=TOL)
        @test isapprox(getvalue(t), getobjectivevalue(eOpt), atol=TOL)
        @test isapprox(getvalue(np), [2.0,1.0,1.0,3.0], atol=TOL)
    end
end

# SDP+Exp problems for conic algorithm with MISOCP
function runsdpexpconicmisocp(mip_solver_drives, mip_solver, cont_solver, log_level)
    @testset "SOC eig cuts: Convex.jl D-opt design" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(dOpt.optval, 9.062207, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,2.0,2.0,1.0], atol=TOL)
    end

    @testset "SOC full cuts: Convex.jl D-opt design" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(dOpt.optval, 9.062207, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,2.0,2.0,1.0], atol=TOL)
    end

    @testset "Init SOC cuts: Convex.jl D-opt design" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(dOpt.optval, 9.062207, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,2.0,2.0,1.0], atol=TOL)
    end

    @testset "Init and eig SOC cuts: Convex.jl D-opt design" begin
        # D-optimal design
        #   maximize    nthroot det V*diag(lambda)*V'
        #   subject to  sum(lambda)=1,  lambda >=0
        n = 7
        nmax = 3
        V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
        (q, p) = size(V)

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
        @test isapprox(dOpt.optval, 9.062207, atol=TOL)
        @test isapprox(Convex.evaluate(Convex.logdet(V * diagm(np./n) * V')), dOpt.optval, atol=TOL)
        @test isapprox(np.value, [2.0,2.0,2.0,1.0], atol=TOL)
    end


end
