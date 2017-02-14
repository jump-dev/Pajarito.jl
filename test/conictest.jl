#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function runconictests(mip_solver_drives, mip_solver, conic_solver, log)
    @testset "Infeasible conic problem" begin
        x = Convex.Variable(1,:Int)

        problem = Convex.maximize(3x,
                            x >= 4,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log), verbose=false)

        @test problem.status == :Infeasible
    end

    @testset "Univariate maximization" begin
        x = Convex.Variable(1,:Int)

        problem = Convex.maximize(3x,
                            x <= 10,
                            x^2 <= 9)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

        @test isapprox(problem.optval, 9.0, atol=TOL)
        @test problem.status == :Optimal
    end

    @testset "Continuous problem" begin
        x = Convex.Variable(1)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       @test_throws ErrorException Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))
    end

    @testset "Maximization problem" begin
        x = Convex.Variable(1,:Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.maximize(3x + y,
                            x >= 0,
                            3x + 2y <= 10,
                            exp(x) <= 10)

       Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

       @test isapprox(problem.optval, 8.0, atol=TOL)
       @test problem.status == :Optimal
    end

    @testset "Solver test" begin
        x = Convex.Variable(1,:Int)
        y = Convex.Variable(1)

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           y >= 0,
                           3x + 2y <= 10,
                           x^2 <= 5,
                           exp(y) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 2.0, atol=TOL)
    end

    @testset "No SOC disaggregation" begin
        x = Convex.Variable(1,:Int)
        y = Convex.Variable(1)

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           y >= 0,
                           3x + 2y <= 10,
                           x^2 <= 5,
                           exp(y) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, soc_disagg=false, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 2.0, atol=TOL)
    end

    @testset "Solver test 2" begin
        x = Convex.Variable(1,:Int)
        y = Convex.Variable(1, Convex.Positive())

        problem = Convex.minimize(-3x - y,
                           x >= 1,
                           3x + 2y <= 30,
                           exp(y^2) + x <= 7)

        Convex.solve!(problem, PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log, rel_gap=1e-4))

        @test problem.status == :Optimal
        @test isapprox(Convex.evaluate(x), 6.0, atol=TOL)
    end

    ## Currently returns UnboundedRelaxation because conic solver interprets infeasible dual incorrectly
    # facts("Conic failure with RSOC - infinite duality gap") do
    #     context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
    #         # Example of polyhedral OA failure due to infinite duality gap from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
    #         # min  z
    #         # st   x == 0
    #         #     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
    #         #      x in {0,1}
    #         m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))
    #         MathProgBase.loadproblem!(m,
    #         [ 0.0, 0.0, 1.0],
    #         [ -1.0  0.0  0.0;
    #          -1.0  0.0  0.0;
    #           0.0 -1.0  0.0;
    #           0.0  0.0 -1.0],
    #         [ 0.0, 0.0, 0.0, 0.0],
    #         Any[(:Zero,1:1),(:SOCRotated,2:4)],
    #         Any[(:Free,[1,2,3])])
    #         MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #         MathProgBase.optimize!(m)
    #         @fact MathProgBase.status(m) --> :ConicFailure
    #    end
    # end

    ## Currently fails on some solver combinations
    # facts("Conic failure with RSOC - finite duality gap") do
    #     context("With $algorithm, $(typeof(mip_solver)) and $(typeof(conic_solver))") do
    #         # Example of polyhedral OA failure due to finite duality gap, modified from "Polyhedral approximation in mixed-integer convex optimization - Lubin et al 2016"
    #         # min  z
    #         # st   x == 0
    #         #     (x,y,z) in RSOC  (2xy >= z^2, x,y >= 0)
    #         #      z >= -10
    #         #      x in {0,1}
    #         m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))
    #         MathProgBase.loadproblem!(m,
    #         [ 0.0, 0.0, 1.0],
    #         [ -1.0  0.0  0.0;
    #          -1.0  0.0  0.0;
    #           0.0 -1.0  0.0;
    #           0.0  0.0 -1.0;
    #           0.0  0.0 -1.0],
    #         [ 0.0, 0.0, 0.0, 0.0, 10.0],
    #         Any[(:Zero,1:1),(:SOCRotated,2:4),(:NonNeg,5:5)],
    #         Any[(:Free,[1,2,3])])
    #         MathProgBase.setvartype!(m, [:Bin,:Cont,:Cont])
    #
    #         MathProgBase.optimize!(m)
    #         @fact MathProgBase.status(m) --> :ConicFailure
    #    end
    # end

    @testset "Variable not in zero cone" begin
        # max  y + z
        # st   x == 1
        #     (x,y,z) in SOC
        #      x in {0,1}
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

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

    @testset "Variable in zero cone" begin
        # Same as "Variable not in zero cone problem" but with variables 2 and 4 added and in zero cones
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

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

    @testset "Rotated SOC problem" begin
        m = MathProgBase.ConicModel(PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=conic_solver, log_level=log))

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
end
