#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#===================================================
 This file implements the default PajaritoSolver
===================================================#

export PajaritoSolver
immutable PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    verbose::Int                # Verbosity level flag
    algorithm                   # Choice of algorithm: "OA" or "BC"
    mip_solver                  # Choice of MILP solver
    cont_solver                 # Choice of Conic solver
    opt_tolerance               # Relatice optimality tolerance
    time_limit                  # Time limit
    profile::Bool               # Performance profile switch
    disaggregate_soc::Symbol    # SOCP disaggregator for SOC constraints
    instance::AbstractString    # Path to instance
    enable_sdp::Bool            # Indicator for enabling sdp support
    force_primal_cuts::Bool     # Enforces primal cutting planes under conic solver
end

function PajaritoSolver(;verbose=0,algorithm="OA",mip_solver=MathProgBase.defaultMIPsolver,cont_solver=MathProgBase.defaultNLPsolver,opt_tolerance=1e-5,time_limit=60*60*10,profile=false,disaggregate_soc=:default,instance="",enable_sdp=false,force_primal_cuts=false)
    disaggregate_soc_ind = Symbol(string(disaggregate_soc))
    PajaritoSolver(verbose,algorithm,mip_solver,cont_solver,opt_tolerance,time_limit,profile,disaggregate_soc_ind,instance,enable_sdp,force_primal_cuts)
end


