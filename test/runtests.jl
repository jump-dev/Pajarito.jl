#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
import ConicBenchmarkUtilities
using Pajarito
using Base.Test


# Tests absolute tolerance and Pajarito printing level
TOL = 1e-3
ll = 3
redirect = true

# Define dictionary of solvers, using JuMP list of available solvers
include(Pkg.dir("JuMP", "test", "solvers.jl"))
include("nlptest.jl")
include("conictest.jl")

solvers = Dict{String,Dict{String,MathProgBase.AbstractMathProgSolver}}()

# MIP solvers
solvers["MILP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["MISOCP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()

tol_int = 1e-9
tol_feas = 1e-8
tol_gap = 0.0

if cpx
    solvers["MILP"]["CPLEX"] = solvers["MISOCP"]["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
    if mos
        solvers["MISOCP"]["Paj(CPLEX+Mosek)"] = PajaritoSolver(mip_solver=CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas/10, CPX_PARAM_EPGAP=tol_gap), cont_solver=Mosek.MosekSolver(LOG=0), log_level=0, rel_gap=1e-7)
    end
end
if glp
    solvers["MILP"]["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas, mip_gap=tol_gap)
    if eco
        solvers["MISOCP"]["Paj(GLPK+ECOS)"] = PajaritoSolver(mip_solver=GLPKMathProgInterface.GLPKSolverMIP(presolve=true, msg_lev=GLPK.MSG_OFF, tol_int=tol_int, tol_bnd=tol_feas/10, mip_gap=tol_gap), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=1e-7)
    end
end
# Gurobi has failed a test due to a known bug; CBC and SCIP fail some tests but the failures have not been fully investigated
# if grb
#     solvers["MILP"]["Gurobi"] = solvers["MISOCP"]["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
# end
#if cbc
#    solvers["MILP"]["CBC"] = Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas, ratioGap=tol_gap, check_warmstart=false)
#    if eco
#        solvers["MISOCP"]["Paj(CBC+ECOS)"] = PajaritoSolver(mip_solver=Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas/10, ratioGap=tol_gap, check_warmstart=false), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=1e-6)
#    end
#end
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
    solvers["PSD+Exp"]["SCS"] = SCS.SCSSolver(eps=1e-5, max_iters=10000000, verbose=0)
    solvers["Exp+SOC"]["SCS"] = SCS.SCSSolver(eps=1e-5, max_iters=10000000, verbose=0)
    solvers["SOC"]["SCS"] = solvers["PSD+SOC"]["SCS"] =  SCS.SCSSolver(eps=1e-6, max_iters=10000000, verbose=0)
end
if mos
    solvers["SOC"]["Mosek"] = solvers["PSD+SOC"]["Mosek"] = Mosek.MosekSolver(LOG=0)
    # Mosek 9+ recognizes the exponential cone:
    # solvers["Exp+SOC"]["Mosek"] = solvers["PSD+Exp"]["Mosek"] = Mosek.MosekSolver(LOG=0)
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
    alg = (msd ? "MSD" : "Iter")

    @testset "MILP solver - $mipname" for (mipname, mip) in solvers["MILP"]
        @testset "NLP models, NLP solver - $conname" for (conname, con) in solvers["NLP"]
            println("\nNLP models, NLP solver: $alg, $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
            run_nlp(msd, mip, con, ll, redirect)
        end

        @testset "LPQP models, SOC solver - $conname" for (conname, con) in solvers["SOC"]
            println("\nLPQP models, SOC solver: $alg, $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
        end

        @testset "Exp+SOC models, NLP solver - $conname" for (conname, con) in solvers["NLP"]
            println("\nExp+SOC models, NLP solver: $alg, $mipname, $conname")
            run_soc(msd, mip, con, ll, redirect)
            run_expsoc(msd, mip, con, ll, redirect)
        end

        @testset "SOC models/solver - $conname" for (conname, con) in solvers["SOC"]
            println("\nSOC models/solver: $alg, $mipname, $conname")
            run_soc(msd, mip, con, ll, redirect)
            run_soc_conic(msd, mip, con, ll, redirect)
        end

        @testset "Exp+SOC models/solver - $conname" for (conname, con) in solvers["Exp+SOC"]
            println("\nExp+SOC models/solver: $alg, $mipname, $conname")
            run_expsoc(msd, mip, con, ll, redirect)
            run_expsoc_conic(msd, mip, con, ll, redirect)
        end

        @testset "PSD+SOC models/solver - $conname" for (conname, con) in solvers["PSD+SOC"]
            println("\nPSD+SOC models/solver: $alg, $mipname, $conname")
            run_sdpsoc_conic(msd, mip, con, ll, redirect)
        end

        @testset "PSD+Exp models/solver - $conname" for (conname, con) in solvers["PSD+Exp"]
            println("\nPSD+Exp models/solver: $alg, $mipname, $conname")
            run_sdpexp_conic(msd, mip, con, ll, redirect)
        end

        flush(STDOUT)
        flush(STDERR)
    end

    @testset "MISOCP solver - $mipname" for (mipname, mip) in solvers["MISOCP"]
        if msd && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip), x -> x)
            # Only test MSD on lazy callback solvers
            continue
        end

        @testset "MISOCP: Exp+SOC models/solver - $conname" for (conname, con) in solvers["Exp+SOC"]
            println("\nMISOCP: Exp+SOC models/solver: $alg, $mipname, $conname")
            run_expsoc_misocp(msd, mip, con, ll, redirect)
        end

        @testset "MISOCP: PSD+SOC solver - $conname" for (conname, con) in solvers["PSD+SOC"]
            println("\nMISOCP: PSD+SOC models/solver: $alg, $mipname, $conname")
            run_sdpsoc_misocp(msd, mip, con, ll, redirect)
        end

        @testset "MISOCP: PSD+Exp solver - $conname" for (conname, con) in solvers["PSD+Exp"]
            println("\nMISOCP: PSD+Exp models/solver: $alg, $mipname, $conname")
            run_sdpexp_misocp(msd, mip, con, ll, redirect)
        end

        flush(STDOUT)
        flush(STDERR)
    end
    println()
end
