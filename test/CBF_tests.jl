# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# CBF instance tests

module TestCBF

using Test
import MathOptInterface
const MOI = MathOptInterface
const MOIFF = MOI.FileFormats
import Pajarito

function runtests(oa_solver, conic_solver)
    @testset "iterative method" begin
        run_cbf_tests(true, oa_solver, conic_solver)
    end
    @testset "one tree method" begin
        run_cbf_tests(false, oa_solver, conic_solver)
    end
    return
end

function run_cbf_tests(use_iter::Bool, oa_solver, conic_solver)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            Pajarito.Optimizer(),
        ),
        Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("use_iterative_method"), use_iter)
    MOI.set(model, MOI.RawOptimizerAttribute("oa_solver"), oa_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("conic_solver"), conic_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("time_limit"), 60)
    MOI.set(model, MOI.RawOptimizerAttribute("iteration_limit"), 100)

    insts = ["sssd_strong_15_4", "exp_ising", "sdp_cardls"]
    folder = joinpath(@__DIR__, "CBF")

    @testset "$inst" for inst in insts
        println(inst)
        file = joinpath(folder, string(inst, ".cbf"))
        run_cbf(model, file)
        getfield(@__MODULE__, Symbol(inst))(model)
    end
    return
end

function run_cbf(model, file::String)
    MOI.empty!(model)
    src = MOIFF.Model(format = MOIFF.FORMAT_CBF)
    MOI.read_from_file(src, file)
    MOI.copy_to(model, src)
    MOI.optimize!(model)
    return model
end

function sssd_strong_15_4(model)
    @test MOI.get(model, MOI.TerminationStatus()) in
          (MOI.TIME_LIMIT, MOI.OPTIMAL)
    return
end

function exp_ising(model)
    TOL = 1e-4
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(model, MOI.ObjectiveValue()), 0.696499, atol = TOL)
    @test isapprox(MOI.get(model, MOI.ObjectiveBound()), 0.696499, atol = TOL)
    return
end

function sdp_cardls(model)
    TOL = 1e-4
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test isapprox(MOI.get(model, MOI.ObjectiveValue()), 16.045564, atol = TOL)
    @test isapprox(MOI.get(model, MOI.ObjectiveBound()), 16.045564, atol = TOL)
    return
end

end
