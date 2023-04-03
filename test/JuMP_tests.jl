# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# JuMP instance tests

module TestJuMP

using Test
import LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Pajarito

function runtests(oa_solver, conic_solver)
    @testset "using conic solver" begin
        @testset "iterative method" begin
            run_jump_tests(true, true, oa_solver, conic_solver)
        end
        @testset "one tree method" begin
            run_jump_tests(false, true, oa_solver, conic_solver)
        end
    end
    @testset "using separation only" begin
        @testset "iterative method" begin
            run_jump_tests(true, false, oa_solver, conic_solver)
        end
        @testset "one tree method" begin
            run_jump_tests(false, false, oa_solver, conic_solver)
        end
    end
    return
end

function run_jump_tests(
    use_iter::Bool,
    solve_relax_subp::Bool,
    oa_solver,
    conic_solver,
)
    opt = JuMP.optimizer_with_attributes(
        Pajarito.Optimizer,
        "verbose" => true,
        # "verbose" => false,
        "use_iterative_method" => use_iter,
        "solve_relaxation" => solve_relax_subp,
        "solve_subproblems" => solve_relax_subp,
        "oa_solver" => oa_solver,
        "conic_solver" => conic_solver,
        "iteration_limit" => 100,
        # "time_limit" => 120.0,
    )
    insts = [
        _soc1,
        _soc2,
        _soc3,
        _exp1,
        _exp2,
        _pow1,
        _pow2,
        _psd1,
        _psd2,
        _expdesign,
        _specialorderedset,
    ]
    @testset "$inst" for inst in insts
        println(inst)
        inst(opt)
    end
    run_cut_test(_soc1, opt)
    run_cut_test(_expdesign, opt)
    return
end

function run_cut_test(test_function, opt)
    MOI.empty!(opt)
    @test MOI.get(opt, Pajarito.NumberOfCuts()) == 0
    test_function(opt)
    @test MOI.get(opt, Pajarito.NumberOfCuts()) > 0
end

function _soc1(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x)
    JuMP.@objective(m, Min, -x)
    xlb1 = JuMP.@constraint(m, x >= 4)
    soc1 = JuMP.@constraint(m, [3.5, x] in JuMP.SecondOrderCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION

    JuMP.delete(m, xlb1)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), -3.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3.5, atol = TOL)
    @test isapprox(JuMP.value(x), 3.5, atol = TOL)

    xlb2 = JuMP.@constraint(m, x >= 3.1)
    JuMP.set_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE

    JuMP.delete(m, xlb2)
    JuMP.@constraint(m, x >= 0.5)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(m), -3, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3, atol = TOL)
    @test isapprox(JuMP.value(x), 3, atol = TOL)

    JuMP.@objective(m, Max, -3x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(m), -3, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3, atol = TOL)
    @test isapprox(JuMP.value(x), 1, atol = TOL)

    JuMP.set_start_value(x, 1)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test isapprox(JuMP.objective_value(m), -3, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -3, atol = TOL)
    @test isapprox(JuMP.value(x), 1, atol = TOL)
    return
end

function _soc2(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x)
    JuMP.@variable(m, y)
    JuMP.@variable(m, z, Int)
    JuMP.@constraint(m, z <= 2.5)
    JuMP.@objective(m, Min, x + 2y)
    JuMP.@constraint(m, [z, x, y] in JuMP.SecondOrderCone())
    JuMP.set_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    opt_obj = -1 - 2 * sqrt(3)
    @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.value(x), -1, atol = TOL)
    @test isapprox(JuMP.value(y), -sqrt(3), atol = TOL)
    @test isapprox(JuMP.value(z), 2, atol = TOL)

    JuMP.unset_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    opt_obj = -2 * sqrt(5)
    @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
    @test isapprox(abs2(JuMP.value(x)) + abs2(JuMP.value(y)), 4, atol = TOL)
    @test isapprox(JuMP.value(z), 2, atol = TOL)
    return
end

function _soc3(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)
    JuMP.@variable(m, x[1:3], Bin)
    # JuMP.set_start_value.(x, 1) # infeasible
    soc = JuMP.@constraint(
        m,
        vcat(inv(sqrt(2)), x .- 0.5) in JuMP.SecondOrderCone()
    )
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION

    JuMP.@objective(m, Min, 1 + sum(x))
    JuMP.delete(m, soc)
    JuMP.@constraint(m, vcat(1.5, x .- 1) in JuMP.SecondOrderCone())
    JuMP.set_start_value.(x, [1, 0, 0])
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 2, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 2, atol = TOL)
    return
end

function _exp1(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    (u, v, w) = s = JuMP.@variable(m, [1:3])
    JuMP.set_binary(v)
    JuMP.@objective(m, Max, u)
    JuMP.@constraint(m, s in MOI.ExponentialCone())
    JuMP.@constraint(m, 1 / 2 - u - v >= 0)
    c1 = JuMP.@constraint(m, w <= 1e5)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 0, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 0, atol = TOL)
    @test isapprox(JuMP.value(v), 0, atol = TOL)

    JuMP.delete(m, c1)
    JuMP.@objective(m, Max, u - w)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 0, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 0, atol = TOL)
    @test isapprox(JuMP.value(w), 0, atol = TOL)
    return
end

function _exp2(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x >= 0, Int)
    JuMP.@variable(m, y >= 0)
    JuMP.@objective(m, Min, -3x - y)
    JuMP.@constraint(m, 3x + 2y <= 10)
    c1 = JuMP.@constraint(m, [x, 1, 10] in MOI.ExponentialCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), -8, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), -8, atol = TOL)
    @test isapprox(JuMP.value(x), 2, atol = TOL)
    @test isapprox(JuMP.value(y), 2, atol = TOL)

    JuMP.delete(m, c1)
    JuMP.@constraint(m, [x, -1, 10] in MOI.ExponentialCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION
    return
end

function _pow1(opt)
    TOL = 1e-3
    m = JuMP.Model(opt)

    JuMP.@variable(m, x)
    JuMP.@variable(m, y)
    JuMP.@variable(m, z)
    JuMP.@constraint(m, z >= 0.5)
    JuMP.@objective(m, Min, x + 2y)
    JuMP.@constraint(m, [x, y, z] in MOI.PowerCone(0.5))
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    rt2 = sqrt(2)
    @test isapprox(JuMP.objective_value(m), rt2, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), rt2, atol = TOL)
    @test isapprox(JuMP.value(x), rt2 / 2, atol = TOL)
    @test isapprox(JuMP.value(y), rt2 / 4, atol = TOL)
    @test isapprox(JuMP.value(z), 0.5, atol = TOL)

    JuMP.set_integer(x)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 1.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 1.5, atol = TOL)
    @test isapprox(JuMP.value(x), 1, atol = TOL)
    @test isapprox(JuMP.value(y), 0.25, atol = TOL)

    JuMP.set_integer(z)
    JuMP.@objective(m, Min, x + 3y)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 3.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 3.5, atol = TOL)
    @test isapprox(JuMP.value(x), 2, atol = TOL)
    @test isapprox(JuMP.value(y), 0.5, atol = TOL)
    @test isapprox(JuMP.value(z), 1, atol = TOL)
    return
end

function _pow2(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)
    JuMP.@variable(m, x[1:3])
    JuMP.@variable(m, y[1:3], Bin)
    JuMP.@constraint(
        m,
        [i in 1:3],
        [x[i] / 3, 1, y[i] - 0.5] in MOI.PowerCone(1 / 3)
    )
    c1 = JuMP.@constraint(m, sum(x) <= 1)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION

    JuMP.delete(m, c1)
    JuMP.@objective(m, Min, sum(x))
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 9 / 8, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 9 / 8, atol = TOL)
    @test isapprox(JuMP.value.(x), fill(3 / 8, 3), atol = TOL)
    y_val = JuMP.value.(y)
    @test isapprox(y_val, round.(y_val), atol = TOL)
    return
end

function _psd1(opt)
    TOL = 1e-4
    m = JuMP.Model(opt)

    JuMP.@variable(m, x, Int)
    JuMP.@constraint(m, x >= 0)
    JuMP.@variable(m, y >= 0)
    JuMP.@variable(m, Z[1:2, 1:2], PSD)
    JuMP.@objective(m, Max, 3x + y - Z[1, 1])
    JuMP.@constraint(m, 3x + 2y <= 10)
    JuMP.@constraint(m, [2 x; x 2] in JuMP.PSDCone())
    c1 = JuMP.@constraint(m, Z[1, 2] >= 1)
    c2 = JuMP.@constraint(m, y >= Z[2, 2])
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 7.5, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 7.5, atol = TOL)
    @test isapprox(JuMP.value(x), 2, atol = TOL)
    @test isapprox(JuMP.value(y), 2, atol = TOL)
    @test isapprox(JuMP.value.(vec(Z)), [0.5, 1, 1, 2], atol = TOL)

    JuMP.delete(m, c1)
    JuMP.delete(m, c2)
    JuMP.@constraint(m, x >= 2)
    JuMP.set_lower_bound(Z[1, 2], 2)
    c2 = JuMP.@constraint(m, y >= Z[2, 2] + Z[1, 1])
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.INFEASIBLE
    @test JuMP.primal_status(m) == MOI.NO_SOLUTION
    return
end

function _psd2(opt)
    TOL = 1e-4
    d = 3
    mat = LinearAlgebra.Symmetric(Matrix{Float64}(reshape(1:(d^2), d, d)), :U)
    λ₁ = LinearAlgebra.eigmax(mat)
    m = JuMP.Model(opt)

    X = JuMP.@variable(m, [1:d, 1:d], PSD)
    JuMP.@objective(m, Max, LinearAlgebra.dot(mat, X))
    JuMP.@constraint(m, LinearAlgebra.tr(X) == 1)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), λ₁, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), λ₁, atol = TOL)
    X_val = JuMP.value.(X)
    @test isapprox(LinearAlgebra.tr(X_val), 1, atol = TOL)

    JuMP.set_binary.(X)
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), mat[d, d], atol = TOL)
    @test isapprox(JuMP.objective_bound(m), mat[d, d], atol = TOL)
    X_val = JuMP.value.(X)
    @test isapprox(sum(X_val), 1, atol = TOL)
    @test isapprox(X_val[d, d], 1, atol = TOL)
    return
end

function _expdesign(opt)
    TOL = 1e-4
    # experiment design
    V = [1 1 -0.2 -0.5; 1 -1 0.5 -0.2]
    function setup_exp_design()
        m = JuMP.Model(opt)
        JuMP.@variable(m, x[1:4], Int)
        JuMP.@constraint(m, x[1:2] .>= 1) # avoids ill-posedness
        JuMP.@constraint(m, x[3:4] .>= 0)
        JuMP.@constraint(m, sum(x) <= 8)
        Q = V * LinearAlgebra.diagm(x) * V'
        return (m, x, Q)
    end

    # A-optimal
    (m, x, Q) = setup_exp_design()
    JuMP.set_start_value.(x, [4, 4, 0, 0]) # partial warm start
    JuMP.@variable(m, y[1:2])
    JuMP.@objective(m, Min, sum(y))
    for i in 1:2
        ei = zeros(2)
        ei[i] = 1
        Qyi = [Q ei; ei' y[i]]
        JuMP.@constraint(m, LinearAlgebra.Symmetric(Qyi) in JuMP.PSDCone())
    end
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 1 / 4, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 1 / 4, atol = TOL)
    x_val = JuMP.value.(x)
    @test isapprox(x_val, [4, 4, 0, 0], atol = TOL)
    @test isapprox(JuMP.value.(y[1]), JuMP.value.(y[2]), atol = TOL)

    # E-optimal
    (m, x, Q) = setup_exp_design()
    JuMP.@variable(m, y)
    JuMP.set_start_value.(vcat(x, y), [4, 4, 0, 0, 8]) # full warm start
    JuMP.@objective(m, Max, y)
    Qy = Q - y * Matrix(LinearAlgebra.I, 2, 2)
    JuMP.@constraint(m, LinearAlgebra.Symmetric(Qy) in JuMP.PSDCone())
    JuMP.optimize!(m)
    @test JuMP.termination_status(m) == MOI.OPTIMAL
    @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
    @test isapprox(JuMP.objective_value(m), 8, atol = TOL)
    @test isapprox(JuMP.objective_bound(m), 8, atol = TOL)
    x_val = JuMP.value.(x)
    @test isapprox(x_val, [4, 4, 0, 0], atol = TOL)

    # D-optimal
    for use_logdet in (true, false)
        opt_x = [4, 4, 0, 0]
        opt_Q = LinearAlgebra.Symmetric(V * LinearAlgebra.Diagonal(opt_x) * V')
        (m, x, Q) = setup_exp_design()
        JuMP.@variable(m, y)
        JuMP.@objective(m, Max, y)
        Qvec = [Q[1, 1], Q[2, 1], Q[2, 2]]
        if use_logdet
            JuMP.@constraint(m, vcat(y, 1.0, Qvec) in MOI.LogDetConeTriangle(2))
            opt_val = LinearAlgebra.logdet(opt_Q)
        else
            JuMP.@constraint(m, vcat(y, Qvec) in MOI.RootDetConeTriangle(2))
            opt_val = sqrt(LinearAlgebra.det(opt_Q))
        end
        JuMP.optimize!(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
        @test isapprox(JuMP.objective_value(m), opt_val, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), opt_val, atol = TOL)
        x_val = JuMP.value.(x)
        @test isapprox(x_val, opt_x, atol = TOL)
    end
    return
end

function _specialorderedset(opt)
    TOL = 1e-4
    for (S, sol) in [(MOI.SOS1, [1, 0, 0]), (MOI.SOS2, [1, 1, 0])]
        if !MOI.supports_constraint(
            MOI.instantiate(opt),
            MOI.VectorOfVariables,
            S{Float64},
        )
            # only test if Pajarito supports SOS1/2
            continue
        end
        m = JuMP.Model(opt)

        JuMP.@variable(m, u)
        JuMP.@variable(m, w[1:3])
        w0 = w + [1.1, 2.3, 3.5]
        JuMP.@constraint(m, vcat(u, w0) in MOI.GeometricMeanCone(4))
        JuMP.@objective(m, Max, u)
        JuMP.@constraint(m, w .<= 1.5)
        JuMP.@constraint(m, w in S([1.0, 2, 3]))
        JuMP.optimize!(m)

        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
        opt_obj = prod(JuMP.value.(w0))^(1 // 3)
        @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
        @test isapprox(JuMP.value(u), opt_obj, atol = TOL)
        @test isapprox(JuMP.value.(w), 1.5 * sol, atol = TOL)

        JuMP.set_integer.(w)
        JuMP.optimize!(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.primal_status(m) == MOI.FEASIBLE_POINT
        opt_obj = prod(JuMP.value.(w0))^(1 // 3)
        @test isapprox(JuMP.objective_value(m), opt_obj, atol = TOL)
        @test isapprox(JuMP.objective_bound(m), opt_obj, atol = TOL)
        @test isapprox(JuMP.value(u), opt_obj, atol = TOL)
        @test isapprox(JuMP.value.(w), sol, atol = TOL)
    end
    return
end

end
