#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Quadratically constrained problems compatible with MathProgBase ConicToLPQPBridge
function run_qp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=(redirect ? 0 : 3))

    @testset "QP optimal" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, Int)
        @variable(m, y >= 0)
        @variable(m, 0 <= u <= 10, Int)
        @variable(m, w == 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 10 <= 20)
        @constraint(m, y^2 <= u*w)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), -12.162277, atol=TOL)
        @test isapprox(getobjbound(m), -12.162277, atol=TOL)
        @test isapprox(getvalue(x), 3, atol=TOL)
        @test isapprox(getvalue(y), 3.162277, atol=TOL)
    end

    @testset "QP maximize" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, Int)
        @variable(m, y >= 0)
        @variable(m, 0 <= u <= 10, Int)
        @variable(m, w == 1)

        @objective(m, Max, 3x + y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @constraint(m, x^2 <= u*w)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), 9.5, atol=TOL)
        @test isapprox(getobjbound(m), 9.5, atol=TOL)
        @test isapprox(getvalue(x), 3, atol=TOL)
        @test isapprox(getvalue(y), 0.5, atol=TOL)
    end

    @testset "QP large (cardls)" begin
        A = [-0.658136 -0.215804 -1.22825 0.636702 0.310855 0.0436465; 0.383753 -0.515882 -1.39494 -0.797658 -0.802035 1.15531; -0.601421 0.15529 0.638735 -0.16043 0.696064 -0.439435; -0.211517 -0.630257 0.614026 -1.4663 1.36299 -0.512717; 1.57874 -0.375281 -0.439124 1.75887 -0.814751 -1.56508; 2.03256 -0.003084 0.573321 -0.874149 -0.148805 0.263757; 0.396071 1.1321 -1.82076 -1.14665 -0.245664 -1.05774; -0.870703 -0.0720246 -0.343017 0.921975 -0.902467 -1.08266; -0.705681 0.180677 1.0088 0.709111 -0.269505 -1.59058; 1.63771 0.524403 0.198447 0.0235749 -1.22018 -1.69565; -0.304213 -0.220045 -0.249271 -0.0956476 -0.860636 0.119479; -0.213992 0.62724 -1.31959 0.907254 0.0394771 1.419; 0.88695 0.43794 0.440619 0.140498 -0.935278 -0.273569; 1.54024 -0.974513 -0.481017 0.41188 -0.211076 -0.618709; -0.134482 1.54252 0.850121 -0.678518 -1.20563 -2.02133; -0.0874732 0.605379 -1.06185 0.0803662 0.00117048 0.507544; -0.414197 -0.627169 -1.49419 -0.677743 0.610031 1.38788; -0.39504 0.025945 -1.36405 0.12975 -0.590624 -0.0804821; 1.31011 1.1715 3.57264 1.24484 1.78609 0.0945148; 1.72996 0.0928935 -0.351372 -1.3813 -0.903951 -0.402878]
        b = [-2.4884, 0.24447, 1.25599, 1.03482, 0.56539, 2.16735, 0.274518, -0.649421, 0.288631, -0.99246, 0.91836, -0.983705, -0.408959, -0.560663, 0.00348301, -0.723511, -0.183856, 0.366346, -1.62336, -0.462939]
        d = 6
        s = 20
        k = floor(Int, d/2)
        rho = 1.
        xB = 4

        m = Model(solver=solver)
        @variable(m, x[1:d])
        @variable(m, z[1:d], Bin)
        @variable(m, u >= 0)
        @variable(m, v >= 0)
        @objective(m, Min, u + rho*v)
        @variable(m, t[1:s])
        @constraint(m, t .== A*x - b)
        @variable(m, w == 2)
        @constraint(m, sum(t.^2) <= u*w)
        @constraint(m, sum(x.^2) <= v*w)
        @constraint(m, x .<= xB.*z)
        @constraint(m, x .>= -xB.*z)
        @constraint(m, sum(z) <= k)

        status = solve(m, suppress_warnings=true)

        @test status == :Optimal
        @test isapprox(getobjectivevalue(m), 8.022766, atol=TOL)
        @test isapprox(getobjbound(m), 8.022766, atol=TOL)
        @test isapprox(getvalue(z), [0, 1, 1, 1, 0, 0], atol=TOL)
    end
end

# NLP problems NOT compatible with MathProgBase ConicToLPQPBridge
function run_nlp(mip_solver_drives, mip_solver, cont_solver, log_level, redirect)
    solver=PajaritoSolver(mip_solver_drives=mip_solver_drives, mip_solver=mip_solver, cont_solver=cont_solver, log_level=(redirect ? 0 : 3))

    @testset "Nonconvex error" begin
        m = Model(solver=solver)

        @variable(m, x >= 0, start = 1, Int)
        @variable(m, y >= 0, start = 1)

        @objective(m, Min, -3x - y)

        @constraint(m, 3x + 2y + 10 <= 20)
        @NLconstraint(m, 8 <= x^2 <= 10)

        @test_throws ErrorException solve(m, suppress_warnings=true)
    end

    @testset "Optimal" begin
        m = Model(solver=solver)

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

    @testset "Infeasible 1" begin
        m = Model(solver=solver)

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
        m = Model(solver=solver)

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

    @testset "Continuous error" begin
        m = Model(solver=solver)

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
        m = Model(solver=solver)

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
        m = Model(solver=solver)

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
