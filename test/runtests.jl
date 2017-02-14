#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import Convex
using Pajarito
using Base.Test

include(Pkg.dir("JuMP", "test", "solvers.jl"))
include("nlptest.jl")
include("conictest.jl")

# Define solvers using JuMP/test/solvers.jl
solvers_mip = lazy_solvers

solvers_nlp = []
if ipt
    push!(solvers_nlp, Ipopt.IpoptSolver(print_level=0))
end
if kni
    push!(solvers_nlp, KNITRO.KnitroSolver(objrange=1e16,outlev=0,maxit=100000))
end

solvers_soc = []
solvers_expsoc = []
solvers_sdpsoc = []
if eco
    push!(solvers_soc, ECOS.ECOSSolver(verbose=false))
    push!(solvers_expsoc, ECOS.ECOSSolver(verbose=false))
end
if scs
    push!(solvers_soc, SCS.SCSSolver(eps=1e-5,max_iters=100000,verbose=0))
    push!(solvers_expsoc, SCS.SCSSolver(eps=1e-5,max_iters=100000,verbose=0))
    push!(solvers_sdpsoc, SCS.SCSSolver(eps=1e-5,max_iters=100000,verbose=0))
end
if mos
    push!(solvers_soc, Mosek.MosekSolver(LOG=0))
    push!(solvers_sdpsoc, Mosek.MosekSolver(LOG=0))
end

println("\nMIP solvers:")
for solver in solvers_mip
    println(solver)
end
println("\nNLP solvers:")
for solver in solvers_nlp
    println(solver)
end
println("\nConic SOC solvers:")
for solver in solvers_soc
    println(solver)
end
println("\nConic Exp+SOC solvers:")
for solver in solvers_expsoc
    println(solver)
end
println("\nConic SDP+SOC solvers:")
for solver in solvers_sdpsoc
    println(solver)
end
println("\nStarting Pajarito tests...\n")

# Tests absolute tolerance and Pajarito printing options
TOL = 1e-3
log = 0

@testset "All Pajarito tests" begin
    # NLP tests in nlptest.jl
    @testset "NLP model and NLP solver tests" begin
        for msd in [false, true], mip in solvers_mip, nlp in solvers_nlp
            @testset "MSD=$msd, MIP=$(typeof(mip)), NLP=$(typeof(nlp))" begin
                runnlptests(msd, mip, nlp, log)
            end
        end
    end

    # Conic models tests in conictest.jl
    @testset "Conic model and NLP solver tests" begin
        @testset "SOC problems" begin
            for msd in [false, true], mip in solvers_mip, nlp in solvers_nlp
                @testset "MSD=$msd, MIP=$(typeof(mip)), NLP=$(typeof(nlp))" begin
                    runsoctests(msd, mip, nlp, log)
                end
            end
        end
        @testset "Exp+SOC problems" begin
            for msd in [false, true], mip in solvers_mip, nlp in solvers_nlp
                @testset "MSD=$msd, MIP=$(typeof(mip)), NLP=$(typeof(nlp))" begin
                    runexpsoctests(msd, mip, nlp, log)
                end
            end
        end
    end

    @testset "Conic model and conic solver tests" begin
        @testset "SOC problems" begin
            for msd in [false, true], mip in solvers_mip, conic in solvers_soc
                @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                    runsoctests(msd, mip, conic, log)
                end
            end
        end
        @testset "Exp+SOC problems" begin
            for msd in [false, true], mip in solvers_mip, conic in solvers_expsoc
                @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                    runexpsoctests(msd, mip, conic, log)
                end
            end
        end
        @testset "SDP+SOC problems" begin
            for msd in [false, true], mip in solvers_mip, conic in solvers_sdpsoc
                @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                    runsdpsoctests(msd, mip, conic, log)
                end
            end
        end
    end
end
