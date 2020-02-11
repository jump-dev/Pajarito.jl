#  Copyright 2017, Chris Coey and Miles Lubin
#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP
using MathProgBase
import ConicBenchmarkUtilities
using Pajarito

using Test
using Printf

using Logging
disable_logging(Logging.Error)

jump_path = joinpath(dirname(pathof(JuMP)), "..")

# Tests absolute tolerance and Pajarito printing level
TOL = 1e-3
ll = 3
redirect = true

# Define dictionary of solvers, using JuMP list of available solvers
include(joinpath(jump_path, "test", "solvers.jl"))
include("qptest.jl")
include("conictest.jl")

solvers = Dict{String,Dict{String,MathProgBase.AbstractMathProgSolver}}()

# MIP solvers
solvers["MILP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["MISOCP"] = Dict{String,MathProgBase.AbstractMathProgSolver}()

tol_int = 1e-9
tol_feas = 1e-7
tol_gap = 0.0

if glp
    solvers["MILP"]["GLPK"] = GLPKMathProgInterface.GLPKSolverMIP(msg_lev=0, tol_int=tol_int, tol_bnd=tol_feas, mip_gap=tol_gap)
    if eco
        solvers["MISOCP"]["Paj(GLPK+ECOS)"] = PajaritoSolver(mip_solver_drives=false, mip_solver=GLPKMathProgInterface.GLPKSolverMIP(presolve=true, msg_lev=0, tol_int=tol_int, tol_bnd=tol_feas/10, mip_gap=tol_gap), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=1e-7)
    end
end
if cpx
    solvers["MILP"]["CPLEX"] = solvers["MISOCP"]["CPLEX"] = CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas, CPX_PARAM_EPGAP=tol_gap)
    if mos
        solvers["MISOCP"]["Paj(CPLEX+Mosek)"] = PajaritoSolver(mip_solver_drives=false, mip_solver=CPLEX.CplexSolver(CPX_PARAM_SCRIND=0, CPX_PARAM_EPINT=tol_int, CPX_PARAM_EPRHS=tol_feas/10, CPX_PARAM_EPGAP=tol_gap), cont_solver=Mosek.MosekSolver(LOG=0), log_level=0, rel_gap=1e-7)
    end
end
if grb
    solvers["MILP"]["Gurobi"] = solvers["MISOCP"]["Gurobi"] = Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas, MIPGap=tol_gap)
    if mos
        solvers["MISOCP"]["Paj(Gurobi+Mosek)"] = PajaritoSolver(mip_solver_drives=false, mip_solver=Gurobi.GurobiSolver(OutputFlag=0, IntFeasTol=tol_int, FeasibilityTol=tol_feas/10., MIPGap=tol_gap), cont_solver=Mosek.MosekSolver(LOG=0), log_level=0, rel_gap=1e-7)
    end
end
#if cbc
#    solvers["MILP"]["CBC"] = Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas, ratioGap=tol_gap, check_warmstart=false)
#    if eco
#        solvers["MISOCP"]["Paj(CBC+ECOS)"] = PajaritoSolver(mip_solver_drives=false, mip_solver=Cbc.CbcSolver(logLevel=0, integerTolerance=tol_int, primalTolerance=tol_feas/10, ratioGap=tol_gap, check_warmstart=false), cont_solver=ECOS.ECOSSolver(verbose=false), log_level=0, rel_gap=1e-6)
#    end
#end
# if try_import(:SCIP)
#     solvers["MILP"]["SCIP"] = solvers["MISOCP"]["SCIP"] = SCIP.SCIPSolver("display/verblevel", 0, "limits/gap", tol_gap, "numerics/feastol", tol_feas)
# end

# Conic solvers
solvers["SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["Exp+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+SOC"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
solvers["PSD+Exp"] = Dict{String,MathProgBase.AbstractMathProgSolver}()
if eco
    solvers["SOC"]["ECOS"] = solvers["Exp+SOC"]["ECOS"] = ECOS.ECOSSolver(verbose=false, reltol=1e-9, feastol=1e-9, reltol_inacc=1e-5, feastol_inacc=1e-8)
end
if scs
    # solvers["SOC"]["SCS"] = solvers["Exp+SOC"]["SCS"] = SCS.SCSSolver(eps=1e-5, max_iters=1e7, verbose=0)
    solvers["PSD+SOC"]["SCS"] = solvers["PSD+Exp"]["SCS"] = SCS.SCSSolver(eps=1e-5, max_iters=1e7, verbose=0)
end
if mos
    solvers["SOC"]["Mosek"] = solvers["PSD+SOC"]["Mosek"] = Mosek.MosekSolver(LOG=0, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-9, MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-10, MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-10, MSK_DPAR_INTPNT_CO_TOL_NEAR_REL=1e3)
    # Mosek 9+ recognizes the exponential cone:
    solvers["Exp+SOC"]["Mosek"] = solvers["PSD+Exp"]["Mosek"] = Mosek.MosekSolver(LOG=0, MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-9, MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-10, MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-10, MSK_DPAR_INTPNT_CO_TOL_NEAR_REL=1e3)
end

println("\nSolvers:")
for (stype, snames) in solvers
    println("\n$stype")
    for (i, sname) in enumerate(keys(snames))
        @printf "%2d  %s\n" i sname
    end
end

@testset "Algorithm - $(msd ? "MSD" : "Iter")" for msd in [false, true]
    alg = (msd ? "MSD" : "Iter")

    @testset "MILP solver - $mipname" for (mipname, mip) in solvers["MILP"]
        if msd && !applicable(MathProgBase.setlazycallback!, MathProgBase.ConicModel(mip), x -> x)
            # Only test MSD on lazy callback solvers
            continue
        end

        @testset "LPQP models, SOC solver - $conname" for (conname, con) in solvers["SOC"]
            println("\nLPQP models, SOC solver: $alg, $mipname, $conname")
            run_qp(msd, mip, con, ll, redirect)
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

        flush(stdout)
        flush(stderr)
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

        flush(stdout)
        flush(stderr)
    end

    println()
end
