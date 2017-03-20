#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import Convex
import ConicBenchmarkUtilities
using Pajarito
using Base.Test


# Tests absolute tolerance and Pajarito printing level
TOL = 1e-3
ll = 0

# Define dictionary of solvers, using JuMP list of available solvers
include(Pkg.dir("JuMP", "test", "solvers.jl"))
include("nlptest.jl")
include("conictest.jl")

solvers = Dict{String,Dict{String,MathProgBase.AbstractMathProgSolver}}()

# MIP solvers
solvers["MILP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["MISOCP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()

tol_int = 1e-8
tol_feas = 1e-7
tol_gap = 1e-8

if grb
    solvers["MILP"]["Gurobi"] = solvers["MISOCP"]["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
end
if cpx
    solvers["MILP"]["CPLEX"] = solvers["MISOCP"]["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
end
if glp
    solvers["MILP"]["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas, tol_obj=tol_gap)
    if eco
        solvers["MISOCP"]["Pajarito(GLPK, ECOS)"] = PajaritoSolver(mip_solver=GLPKMathProgInterface.GLPKSolverMIP(presolve=true, msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas, tol_obj=tol_gap), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=tol_gap)
    end
end
# if try_import(:SCIP)
#     solvers["MILP"]["SCIP"] = solvers["MISOCP"]["SCIP"] = SCIP.SCIPSolver("display/verblevel", 0, "limits/gap", tol_gap, "numerics/feastol", tol_feas)
# end

# NLP solvers
solvers["NLP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if ipt
    solvers["NLP"]["Ipopt"] = Ipopt.IpoptSolver(print_level=0)
end
# if kni
#     solvers["NLP"]["Knitro"] = KNITRO.KnitroSolver(objrange=1e16, outlev=0, maxit=100000)
# end

# Conic solvers
solvers["SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["Exp+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+Exp"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if eco
    solvers["SOC"]["ECOS"] = solvers["Exp+SOC"]["ECOS"] = ECOS.ECOSSolver(verbose=false)
end
if scs
    solvers["Exp+SOC"]["SCS"] = SCS.SCSSolver(eps=1e-5, max_iters=1000000, verbose=0)
    solvers["SOC"]["SCS"] = solvers["PSD+SOC"]["SCS"] = solvers["PSD+Exp"]["SCS"] = SCS.SCSSolver(eps=1e-6, max_iters=1000000, verbose=0)
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
        if !(msd && mipname == "GLPK") # GLPK MSD is broken
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
