# Copyright (c) 2021-2022 Chris Coey and contributors
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# all tests

import Test
import MathOptInterface
const MOI = MathOptInterface

import GLPK
glpk = MOI.OptimizerWithAttributes(
    GLPK.Optimizer,
    MOI.Silent() => true,
    "tol_int" => 1e-10,
    "tol_bnd" => 1e-10,
    "mip_gap" => 1e-10,
)

import Hypatia
hypatia = MOI.OptimizerWithAttributes(
    Hypatia.Optimizer,
    MOI.Silent() => true,
    "near_factor" => 100,
    "tol_feas" => 1e-8,
    "tol_rel_opt" => 1e-8,
    "tol_abs_opt" => 1e-8,
    "tol_illposed" => 1e-8,
    "tol_slow" => 2e-2,
    "tol_inconsistent" => 1e-7,
)

println("starting Pajarito tests")
Test.@testset "Pajarito tests" begin
    println("starting MOI tests")
    include("MOI_tests.jl")
    Test.@testset "MOI tests" begin
        TestMOI.runtests(glpk, hypatia)
    end

    println("starting JuMP tests")
    include("JuMP_tests.jl")
    Test.@testset "JuMP tests" begin
        TestJuMP.runtests(glpk, hypatia)
    end

    println("starting CBF tests")
    include("CBF_tests.jl")
    Test.@testset "CBF tests" begin
        TestCBF.runtests(glpk, hypatia)
    end
end
