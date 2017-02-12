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
solvers_nlnr = []
ipt && push!(solvers_nlnr, Ipopt.IpoptSolver(print_level=0))
#kni && push!(solvers_nlnr, KNITRO.KnitroSolver(objrange=1e16,outlev=0,maxit=100000))
solvers_conic = eco ? Any[ECOS.ECOSSolver(verbose=false)] : []
solvers_sdp = mos ? Any[Mosek.MosekSolver(LOG=0)] : []

if scs
    push!(solvers_conic,SCS.SCSSolver(eps=1e-5,max_iters=1000000,verbose=0))
    push!(solvers_sdp,SCS.SCSSolver(eps=1e-5,max_iters=1000000,verbose=0))
end
@show solvers_mip
@show solvers_nlnr
@show solvers_conic
@show solvers_sdp

TOL = 1e-3

# Option to print with log_level
log = 2

# Nonlinear models tests in nlptest.jl
# @testset "Nonlinear tests" begin
#     for mip_solver_drives in [false, true], mip in solvers_mip, nlnr in solvers_nlnr
#         @testset "MSD=$mip_solver_drives,mip_solver=$(typeof(mip)),cont_solver=$(typeof(nlnr))" begin
#             runnonlineartests(mip_solver_drives, mip, nlnr, log)
#         end
#     end
# end

# Conic models test in conictest.jl
# Default solvers test
runconicdefaulttests(false, log)
@testset "Conic tests" begin
    for mip_solver_drives in [false, true], mip in solvers_mip
        # Conic model with conic solvers
        for conic in solvers_conic
            @testset "MSD=$mip_solver_drives,mip_solver=$(typeof(mip)),cont_solver=$(typeof(conic))" begin
                runconictests(mip_solver_drives, mip, conic, log)
            end
        end

        # Conic model with nonlinear solvers
        for nlnr in solvers_nlnr
            @testset "MSD=$mip_solver_drives,mip_solver=$(typeof(mip)),cont_solver=$(typeof(nlnr))" begin
                runconictests(mip_solver_drives, mip, nlnr, log)
            end
        end
    end
end

# SDP conic models tests in sdptest.jl
@testset "SDP tests" begin
    for mip_solver_drives in [false, true], mip in solvers_mip, sdp in solvers_sdp
        @testset "SDP tests (MSD=$mip_solver_drives,mip_solver=$(typeof(mip)),cont_solver=$(sdp)" begin
            runsdptests(mip_solver_drives, mip, sdp, log)
        end
    end
end
