#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import Convex
import ConicBenchmarkUtilities
using Pajarito
using Base.Test


# Tests absolute tolerance and Pajarito printing options
TOL = 1e-3
ll = 0


# Define dictionary of solvers, using JuMP list of available solvers
include(Pkg.dir("JuMP", "test", "solvers.jl"))
include("nlptest.jl")
include("conictest.jl")

solvers = Dict{String,Dict{String,MathProgBase.AbstractMathProgSolver}}()

solvers["MILP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["MISOCP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if grb
    solvers["MILP"]["Gurobi"] = solvers["MISOCP"]["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=1e-8, FeasibilityTol=1e-7, MIPGap=1e-8)
end
if cpx
    solvers["MILP"]["CPLEX"] = solvers["MISOCP"]["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=1e-8, CPX_PARAM_EPRHS=1e-7, CPX_PARAM_EPGAP=1e-8)
end
if glp
    solvers["MILP"]["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=1e-8, tol_bnd=1e-7, tol_obj=1e-8)
    if eco
        solvers["MISOCP"]["Pajarito(GLPK, ECOS)"] = PajaritoSolver(mip_solver=GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=1e-8, tol_bnd=1e-7, tol_obj=1e-8), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=1e-8)
    end
end

solvers["NLP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if ipt
    solvers["NLP"]["Ipopt"] = Ipopt.IpoptSolver(print_level=0)
end
# if kni
#     solvers["NLP"]["Knitro"] = KNITRO.KnitroSolver(objrange=1e16, outlev=0, maxit=100000)
# end

solvers["SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["Exp+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+Exp"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if eco
    solvers["SOC"]["ECOS"] = solvers["Exp+SOC"]["ECOS"] = ECOS.ECOSSolver(verbose=false)
end
if scs
    solvers["SOC"]["SCS"] = solvers["Exp+SOC"]["SCS"] = solvers["PSD+SOC"]["SCS"] = solvers["PSD+Exp"]["SCS"] = SCS.SCSSolver(eps=1e-6, max_iters=1000000, verbose=0)
end
if mos
    solvers["SOC"]["Mosek"] = solvers["PSD+SOC"]["Mosek"] = Mosek.MosekSolver(LOG=0)
end


println("\nSolvers:")
for (stype, snames) in solvers
    println("\n$stype")
    for (i, sname) in enumerate(keys(snames))
        @printf "%2d  %s\n" i sname
    end
end
println()


@testset "Algorithm - $(msd ? "MSD" : "Iter")" for msd in [false, true]
    @testset "MILP solver - $mipname" for (mipname, mip) in solvers["MILP"]
        @testset "NLP solver - $conname" for (conname, con) in solvers["NLP"]
            runnlp(msd, mip, con, ll)
            runsocnlpconic(msd, mip, con, ll)
            runexpsocnlpconic(msd, mip, con, ll)
            flush(STDOUT)
        end

        @testset "SOC solver - $conname" for (conname, con) in solvers["SOC"]
            runsocnlpconic(msd, mip, con, ll)
            runsocconic(msd, mip, con, ll)
            flush(STDOUT)
        end

        @testset "Exp+SOC solver - $conname" for (conname, con) in solvers["Exp+SOC"]
            runexpsocnlpconic(msd, mip, con, ll)
            runexpsocconic(msd, mip, con, ll)
            flush(STDOUT)
        end

        @testset "PSD+SOC solver - $conname" for (conname, con) in solvers["PSD+SOC"]
            runsdpsocconic(msd, mip, con, ll)
            flush(STDOUT)
        end

        @testset "PSD+Exp solver - $conname" for (conname, con) in solvers["PSD+Exp"]
            runsdpexpconic(msd, mip, con, ll)
            flush(STDOUT)
        end
    end

    @testset "MISOCP solver - $mipname" for (mipname, mip) in solvers["MISOCP"]
        if !msd || applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip), _ -> _)
            @testset "Exp+SOC solver - $conname" for (conname, con) in solvers["Exp+SOC"]
                runexpsocconicmisocp(msd, mip, con, ll)
                flush(STDOUT)
            end

            @testset "PSD+SOC solver - $conname" for (conname, con) in solvers["PSD+SOC"]
                runsdpsocconicmisocp(msd, mip, con, ll)
                flush(STDOUT)
            end

            @testset "PSD+Exp solver - $conname" for (conname, con) in solvers["PSD+Exp"]
                runsdpexpconicmisocp(msd, mip, con, ll)
                flush(STDOUT)
            end
        end
    end
end
