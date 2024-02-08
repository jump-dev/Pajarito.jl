# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

module TestMOI

using Test

import MathOptInterface as MOI
import Pajarito

function runtests(oa_solver, conic_solver)
    @testset "solving conic subproblems" begin
        @testset "iterative method" begin
            run_moi_tests(true, true, oa_solver, conic_solver)
        end
        @testset "one tree method" begin
            run_moi_tests(false, true, oa_solver, conic_solver)
        end
    end
    @testset "not solving conic subproblems" begin
        @testset "iterative method" begin
            run_moi_tests(true, false, oa_solver, conic_solver)
        end
        @testset "one tree method" begin
            run_moi_tests(false, false, oa_solver, conic_solver)
        end
    end
    return
end

function run_moi_tests(
    use_iter::Bool,
    solve_subp::Bool,
    oa_solver,
    conic_solver,
)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            Pajarito.Optimizer(),
        ),
        Float64,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("use_iterative_method"), use_iter)
    MOI.set(model, MOI.RawOptimizerAttribute("solve_subproblems"), solve_subp)
    MOI.set(model, MOI.RawOptimizerAttribute("oa_solver"), oa_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("conic_solver"), conic_solver)
    MOI.set(model, MOI.RawOptimizerAttribute("iteration_limit"), 100)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            Float64,
            atol = 2e-4,
            rtol = 2e-4,
            exclude = Any[
                MOI.ConstraintDual,
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.DualObjectiveValue,
                MOI.SolverVersion,
            ],
        ),
        exclude = String[
            # TODO: unexpected failures, probably in the bridge layer
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",
            # TODO: errors bridging to MILP with GLPK
            "_Indicator_",
            "_SOS1_",
            "_SOS2_",
        ],
    )
    return
end

end
