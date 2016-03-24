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
    cut_switch                  # Cut level for OA
    socp_disaggregator::Bool    # SOCP disaggregator for SOC constraints
    instance::AbstractString    # Path to instance
end
PajaritoSolver(;verbose=0,algorithm="OA",mip_solver=MathProgBase.defaultMIPsolver,cont_solver=MathProgBase.defaultNLPsolver,opt_tolerance=1e-5,time_limit=60*60*10,cut_switch=1,socp_disaggregator=false,instance="") = PajaritoSolver(verbose,algorithm,mip_solver,cont_solver,opt_tolerance,time_limit,cut_switch,socp_disaggregator,instance)


