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
include("sdptest.jl")

# Define solvers using JuMP/test/solvers.jl
solvers_mip = lazy_solvers

solvers_nlp = []
ipt && push!(solvers_nlp, Ipopt.IpoptSolver(print_level=0))
#kni && push!(solvers_nlp, KNITRO.KnitroSolver(objrange=1e16,outlev=0,maxit=100000))

solvers_conic = eco ? Any[ECOS.ECOSSolver(verbose=false)] : []
solvers_sdp = mos ? Any[Mosek.MosekSolver(LOG=0)] : []
if scs
    push!(solvers_conic, SCS.SCSSolver(eps=1e-5,max_iters=100000,verbose=0))
    push!(solvers_sdp, SCS.SCSSolver(eps=1e-5,max_iters=100000,verbose=0))
end

@show solvers_mip
@show solvers_nlp
@show solvers_conic
@show solvers_sdp

TOL = 1e-3

# Option to print with log_level
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
            runconictests(msd, mip, nlp, log)
        end
    end
end
@testset "Conic model and conic solver tests" begin
    for msd in [false, true], mip in solvers_mip, conic in solvers_conic
        @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(conic))" begin
            runconictests(msd, mip, conic, log)
        end
    end
end

# SDP conic models tests in sdptest.jl
@testset "SDP conic model/solver tests" begin
    for msd in [false, true], mip in solvers_mip, sdp in solvers_sdp
        @testset "MSD=$msd, MIP=$(typeof(mip)), Conic=$(typeof(sdp))" begin
            runsdptests(msd, mip, sdp, log)
        end
    end
end
