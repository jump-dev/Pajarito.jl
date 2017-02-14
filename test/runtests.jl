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
@show solvers_mip
println("\nNLP solvers:")
@show solvers_nlp
println("\nConic SOC solvers:")
@show solvers_soc
println("\nConic Exp+SOC solvers:")
@show solvers_expsoc
println("\nConic SDP+SOC solvers:")
@show solvers_sdpsoc
println()

# Tests absolute tolerance and Pajarito printing options
TOL = 1e-3
log = 2

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
    for msd in [false, true], mip in solvers_mip, nlp in solvers_nlp
        @testset "MSD=$msd, MIP=$(typeof(mip)), NLP=$(typeof(nlp))" begin
            runsoctests(msd, mip, nlp, log)
            runexpsoctests(msd, mip, nlp, log)
        end
    end
end

@testset "Conic model and conic solver tests" begin
    @testset "SOC problems"
        for msd in [false, true], mip in solvers_mip, conic in solvers_soc
            @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                runsoctests(msd, mip, conic, log)
            end
        end
    end
    @testset "Exp+SOC problems"
        for msd in [false, true], mip in solvers_mip, conic in solvers_expsoc
            @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                runexpsoctests(msd, mip, conic, log)
            end
        end
    end
    @testset "SDP+SOC problems"
        for msd in [false, true], mip in solvers_mip, conic in solvers_sdpsoc
            @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
                runsdpsoctests(msd, mip, conic, log)
            end
        end
    end
end
