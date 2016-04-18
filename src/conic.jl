#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#=========================================================
 This solver implements mixed-integer conic programming
 algorithm found in:

  Lubin, Yamangil, Bent, Vielma (2016), Extended formulations
  in Mixed-integer Convex Programming, IPCO 2016, Liege, Belgium
  (available online at http://arxiv.org/abs/1511.06710).

 The package accepts a problem of the form:
  
  min  c_x^T x + c_z^T z
  s.t. A_x x + A_z z = b
       L <= x <= U
       x \in Z
       z \in K

 where K = K_1 \times ... \times K_l is a product of
 simple cones, i.e. second-order cone, exponential cone,
 non-negative cone, free cone.

 To access the solver, one can either:
  1. Create a JuMP.jl model using a conic input format
  2. Create a Convex.jl model using a set of disciplined
     convex atoms defined in Convex.jl library.
 Both ways PajaritoSolver must be provided as the
 preferred solver with a corresponding mip_solver and 
 cont_solver specified.
=========================================================#

using JuMP

type PajaritoConicModel <: MathProgBase.AbstractConicModel
    # SOLUTION DATA:
    solution::Vector{Float64}   # Vector containing solution
    status                      # Termination status of algorithm
    objval::Float64             # Best found objective corresponding to solution
    iterations::Int             # Number of outer approximation iterations if algorithm is OA

    # SOLVER DATA:
    verbose::Int                # Verbosity level flag
    algorithm                   # Choice of algorithm: "OA" or "BC"
    mip_solver                  # Choice of MILP solver
    cont_solver                 # Choice of Conic solver
    opt_tolerance               # Relative optimality tolerance
    time_limit                  # Time limit
    profile::Bool               # Performance profile switch
    disaggregate_soc::DisaggSOC # Disaggregate SOC constraints following Vielma et al.
    instance::AbstractString    # Path to instance
    enable_sdp::Bool            # Indicator for enabling sdp support
    force_primal_cuts::Bool     # Enforces primal cutting planes under conic solver
    dual_cut_zero_tol::Float64  # Tolerance to check if a conic_dual is zero or not

    # PROBLEM DATA
    numVar::Int                 # Number of variables
    numVar_ini::Int             # Number of initial variables
    numIntVar::Int              # Number of integer or binary variables
    numConstr::Int              # Number of constraints
    c                           # Objective coefficients           
    A                           # Affine constraint matrix
    b                           # Affine constraint right hand side
    c_ini                       
    A_ini
    b_ini
    constr_cones_ini
    var_cones_ini
    mip_x                       # Original variables for the MILP model
    mip_t                       # SOCP disaggregator variables if disaggregate_soc is true
    l::Vector{Float64}          # Variable lower bounds
    u::Vector{Float64}          # Variable upper bounds
    vartype::Vector{Symbol}     # Vector containing variable types, :Bin, :Cont, :Int
    objsense::Symbol            # Sense of the objective
    numSpecCones                # Number of special cones (i.e. SOC, Exp) in the problem
    lengthSpecCones             # Dimension of special cones
    numSOCCones                 # Number of SOC cones
    dimSOCCones                 # (Dimension-1) of SOC cones
    pajarito_var_cones          # Variable cones
    pajarito_constr_cones       # Constraint cones
    problem_type
    is_conic_solver             # Indicator if subproblem solver is true conic solver

    # TIMERS FOR PROFILING
    remove_timer
    prep_timer
    nlp_load_timer

    # CONSTRUCTOR:
    function PajaritoConicModel(verbose,algorithm,mip_solver,cont_solver,opt_tolerance,time_limit,profile,disaggregate_soc,instance,enable_sdp,force_primal_cuts,dual_cut_zero_tol)
        m = new()
        m.verbose = verbose
        m.algorithm = algorithm
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.opt_tolerance = opt_tolerance
        m.time_limit = time_limit
        m.profile = profile
        m.disaggregate_soc = disaggregate_soc
        m.instance = instance
        m.enable_sdp = enable_sdp
        m.force_primal_cuts = force_primal_cuts
        m.dual_cut_zero_tol = dual_cut_zero_tol
        return m
    end
end

# BEGIN MATHPROGBASE INTERFACE
MathProgBase.ConicModel(s::PajaritoSolver) = PajaritoConicModel(s.verbose, s.algorithm, s.mip_solver, s.cont_solver, s.opt_tolerance, s.time_limit, s.profile, s.disaggregate_soc, s.instance, s.enable_sdp, s.force_primal_cuts, s.dual_cut_zero_tol)

function MathProgBase.loadproblem!(
    m::PajaritoConicModel, c, A, b, constr_cones, var_cones)

    # Check if the cont_solver is a conic solver
    m.is_conic_solver = (applicable(MathProgBase.ConicModel, m.cont_solver) && m.cont_solver != MathProgBase.defaultNLPsolver)

    # Keep default soc disaggregate only if conic solver
    m.disaggregate_soc = (m.disaggregate_soc == _disagg_soc_default ? (m.is_conic_solver ? _disagg_soc_on : _disagg_soc_off) : m.disaggregate_soc)

    # Wrap nonlinear solver with ConicNonlinearBridge
    m.cont_solver = (m.is_conic_solver ? m.cont_solver : ConicNLPWrapper(nlp_solver=m.cont_solver))

    m.c_ini = c
    m.A_ini = A
    m.b_ini = b
    m.constr_cones_ini = constr_cones
    m.var_cones_ini = var_cones

    numVar = length(c) # number of variables
    numConstr = length(b) # number of constraints

    # b - Ax \in K => b - Ax = s, s \in K
    pajarito_var_cones = Any[x for x in var_cones]
    pajarito_constr_cones = Any[]
    copy_constr_cones = copy(constr_cones)
    lengthSpecCones = 0
    # ADD SLACKS FOR ONLY SOC AND EXP
    A_I, A_J, A_V = findnz(A)
    slack_count = numVar+1
    for (cone, ind) in copy_constr_cones
        if cone == :SDP && !m.enable_sdp
            error("MISDP feature is currently experimental, turn it on by using ""enable_sdp=true"" at your own risk!")
        end
        if cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP
            lengthSpecCones += length(ind)
            slack_vars = slack_count:(slack_count+length(ind)-1)
            append!(A_I, ind)
            append!(A_J, slack_vars)
            append!(A_V, ones(length(ind)))
            push!(pajarito_var_cones, (cone, slack_vars))
            push!(pajarito_constr_cones, (:Zero, ind))
            slack_count += length(ind)
        else
            push!(pajarito_constr_cones, (cone, ind))
        end
    end
    A = sparse(A_I,A_J,A_V, numConstr, numVar+lengthSpecCones)

    m.numVar_ini = numVar
    m.numVar = size(A,2)
    @assert m.numVar == numVar + lengthSpecCones
    m.numConstr = numConstr
    m.c = [c;zeros(m.numVar-numVar)]
    m.A = A
    m.b = b   
    # figure out variable bounds for MIP
    l = -Inf*ones(m.numVar)
    u = Inf*ones(m.numVar)
    numSpecCones = 0
    numSOCCones = 0
    dimSOCCones = Any[]
    soc_indicator = false
    exp_indicator = false
    for (cone, ind) in pajarito_var_cones
        if cone == :SDP && !m.enable_sdp
            error("MISDP feature is currently experimental, turn it on by using ""enable_sdp=true"" at your own risk!")
        end
        if cone == :Free
            # do nothing
        elseif cone == :Zero
            l[ind] = 0.0
            u[ind] = 0.0
        elseif cone == :NonNeg
            l[ind] = 0.0
        elseif cone == :NonPos
            u[ind] = 0.0
        end
        if cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP
            if cone == :SOC
                soc_indicator = true
                l[ind[1]] = 0.0
                numSOCCones += 1
                push!(dimSOCCones,length(ind)-1)
            elseif cone == :SOCRotated
                l[ind[1]] = 0.0
                l[ind[2]] = 0.0
            else
                exp_indicator = true
                l[ind[end]] = 0.0
                l[ind[2]] = 0.0
            end
            numSpecCones += 1
        end
    end
    if soc_indicator && exp_indicator
        problem_type = "ExpSOC"
    elseif soc_indicator
        problem_type = "SOC"
    elseif exp_indicator
        problem_type = "Exp"
    else
        problem_type = "None"
    end

    m.l = l
    m.u = u
    m.numSpecCones = numSpecCones
    m.lengthSpecCones = lengthSpecCones
    m.numSOCCones = numSOCCones
    m.dimSOCCones = dimSOCCones
    m.pajarito_var_cones = pajarito_var_cones
    m.pajarito_constr_cones = pajarito_constr_cones
    m.problem_type = problem_type
    m.vartype = fill(:Cont,m.numVar)
    m.solution = fill(NaN, m.numVar)
end


function addSlackValues(m::PajaritoConicModel, separator)
    rhs_value = m.b_ini - m.A_ini * separator[1:m.numVar_ini]
    slack_count = 1
    separator_slack = zeros(m.lengthSpecCones)
    for (cone, ind) in m.constr_cones_ini
        if cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP
            slack_vars = slack_count:(slack_count+length(ind)-1)
            separator_slack[slack_vars] = rhs_value[ind]
            slack_count += length(ind)
        end
    end
    @assert slack_count == m.lengthSpecCones+1
    return [separator[1:m.numVar_ini];separator_slack]
end

function preprocessIntegersOut(m, c_ini, A_ini, b_ini, mip_solution, vartype, var_cones, numVar)

    start = time()

    c_new = copy(c_ini)
    A_new = copy(A_ini)
    b_new = copy(b_ini)

    k = 1
    removableColumnIndicator = [false for i in 1:numVar]
    new_variable_index_map = -ones(Int, numVar)
    old_variable_index_map = zeros(Int, numVar - m.numIntVar)
    for i in 1:numVar
        if vartype[i] == :Int || vartype[i] == :Bin
            removableColumnIndicator[i] = true
        else
            new_variable_index_map[i] = k
            old_variable_index_map[k] = i
            k += 1
        end   
    end
    new_var_cones = Any[]
    for (cone, ind) in var_cones
        new_ind = Int[]
        for i in ind
            # THIS ASSUMES INTEGER VARIABLES DO NOT APPEAR IN
            # SPEC CONES
            if (cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP) && (vartype[i] == :Int || vartype[i] == :Bin)
                error("Integer variable x[$i] inside $cone cone")
            end
            if new_variable_index_map[i] != -1
                push!(new_ind, new_variable_index_map[i])
            end
        end
        push!(new_var_cones, (cone, new_ind))
    end

    c_new = c_new[!removableColumnIndicator]
    b_new = b_new - A_new[:,removableColumnIndicator]*mip_solution[removableColumnIndicator]
    A_new = A_new[:,!removableColumnIndicator]

    # TODO the following is questionable to do, but it reduced duality gap at expDesigns
    #zero_ind = filter(i->abs(b_new[i]) < 1e-8, 1:m.numConstr)
    #b_new[zero_ind] = zeros(length(zero_ind))

    m.prep_timer += time() - start

    return c_new, A_new, b_new, new_var_cones, old_variable_index_map, new_variable_index_map

end

function removeRedundantRows(m,constr_cones, A_new, b_new)

    start = time()

    @assert size(A_new, 1) == length(b_new)
    (numConstr,numVar) = size(A_new)
    emptyRow = [false for i in 1:numConstr]
    numFullRows = 0
    for i = 1:numConstr
        emptyRowInd = true
        for j = 1:numVar
            if abs(A_new[i,j]) >= 1e-5
                emptyRowInd = false
                numFullRows += 1
                break
            end
        end
        emptyRow[i] = emptyRowInd
    end

    k = 1
    new_constraint_index_map = -ones(Int,numConstr)
    old_constraint_index_map = zeros(Int, numFullRows)
    for i in 1:numConstr
        if !emptyRow[i]
            new_constraint_index_map[i] = k
            old_constraint_index_map[k] = i
            k += 1
        end   
    end

    new_constr_cones = Any[]
    for (cone, ind) in constr_cones
        new_ind = Int[]
        for i in ind
            if (cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP) && (emptyRow[i])
                error("Empty row $i inside $cone cone")
            end
            if new_constraint_index_map[i] != -1
                push!(new_ind, new_constraint_index_map[i])
            end
        end
        push!(new_constr_cones, (cone, new_ind))
    end

    for (cone, ind) in constr_cones
        for i in ind
            if emptyRow[i]
                if cone == :Zero && abs(b_new[i]) > 1e-5
                    error("Infeasible problem 0.0 == $(b_new[i])")
                elseif cone == :NonNeg && b_new[i] < -1e-5
                    error("Infeasible problem 0.0 <= $(b_new[i])")
                elseif cone == :NonPos && b_new[i] > 1e-5
                    error("Infeasible problem 0.0 >= $(b_new[i])")
                elseif cone != :Zero && cone != :NonNeg && cone != :NonPos
                    error("Unrecognized cone $cone")
                end
            end
        end
    end

    m.remove_timer += time() - start

    return new_constr_cones, A_new[!emptyRow,:], b_new[!emptyRow]
end

function loadFirstPhaseConicModel(m::PajaritoConicModel, inf_dcp_model, mip_solution)
    
    # original variables; slack variables; simplification variables
    c_new = [zeros(m.numVar);ones(m.numSpecCones);zeros(m.numSpecCones)]

    (I, J, V) = findnz(m.A)
    b_new = copy(m.b)
    inf_var_cones = Any[]
    k = 1

    inf_constr_cones = copy(m.pajarito_constr_cones)

    for (cone, ind) in m.pajarito_var_cones
        # ADD SOC RELAXATION
        # (y, x) \in SOC => y \geq || x ||
        #   => y + s \geq || x ||
        #   => q = y + s, q \geq || x ||
        #   => q = y + s, (q, x) \in SOC, MIN s
        if cone == :SOC
            ind0 = ind[1]
            push!(I, k + m.numConstr)
            push!(J, ind0)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar + m.numSpecCones)
            push!(V, -1.0)
            push!(b_new, 0.0)
            push!(inf_var_cones, (:SOC, [k + m.numVar + m.numSpecCones; ind[2:end]]))
            push!(inf_var_cones, (:NonNeg, [ind0; k+m.numVar]))
            k += 1       
        # ADD ROTATED SOC RELAXATION
        # (y, z, x) \in SOC => 2yz \geq || x ||^2
        #   => 2y + 2s \geq || x ||^2/z
        #   => q = y + s, 2qz \geq || x ||^2
        #   => q = y + s, (q, z, x) \in ROTATED SOC, MIN s
        elseif cone == :SOCRotated
            ind0 = ind[1]
            push!(I, k + m.numConstr)
            push!(J, ind0)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar + m.numSpecCones)
            push!(V, -1.0)
            push!(b_new, 0.0)
            push!(inf_var_cones, (:SOCRotated, [k + m.numVar + m.numSpecCones; ind[2:end]]))
            push!(inf_var_cones, (:NonNeg, [ind0; k+m.numVar]))
            k += 1     
        # ADD EXP RELAXATION
        # {(x,y,z)∈ ℝ 3:y>0,ye^(x/y) ≤ z}
        #   => z + s \geq y exp(x/y)
        #   => q = z + s, q \geq y exp(x/y)
        #   => q = z + s, (x,y,q) \in ExpPrimal, MIN s
        elseif cone == :ExpPrimal
            ind0 = ind[end]
            push!(I, k + m.numConstr)
            push!(J, ind0)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar)
            push!(V, 1.0)
            push!(I, k + m.numConstr)
            push!(J, k + m.numVar + m.numSpecCones)
            push!(V, -1.0)
            push!(b_new, 0.0)
            push!(inf_var_cones, (:ExpPrimal, [ind[1];ind[2];k + m.numVar + m.numSpecCones]))
            push!(inf_var_cones, (:NonNeg, [ind0; k+m.numVar]))
            k += 1 
        else
            push!(inf_var_cones, (cone,ind))
        end
    end
    @assert k == m.numSpecCones+1
    # FIX BINARY OR INTEGER VARIABLES TO CURRENT MIP SOLUTION
    # ONLY FOR ADDING EQUALITY CONSTRAINTS
    k = 1
    for i in 1:m.numVar
        if m.vartype[i] == :Int || m.vartype[i] == :Bin
            push!(I, m.numConstr + m.numSpecCones + k)
            push!(J, i)
            push!(V, 1.0)
            push!(b_new, round(mip_solution[i]))
            k += 1
        end
    end
    inf_constr_cones = [inf_constr_cones;(:Zero, (m.numConstr+1):(length(b_new)))]

    A_new = sparse(I,J,V)
    # PREPROCESS THEM OUT OF MATRIX
    extended_mip_solution = [mip_solution;zeros(2*m.numSpecCones)]
    vartype = [m.vartype;[:Cont for i in 1:2*m.numSpecCones]]
    infNumVar = m.numVar + 2*m.numSpecCones

    (c_new, A_new, b_new, new_var_cones, old_variable_index_map, new_variable_index_map) = 
        preprocessIntegersOut(m, c_new, A_new, b_new, extended_mip_solution, vartype, inf_var_cones, infNumVar)

    (new_constr_cones, A_new, b_new) = removeRedundantRows(m, inf_constr_cones, A_new, b_new)

    start = time()
    MathProgBase.loadproblem!(inf_dcp_model, c_new, A_new, b_new, new_constr_cones, new_var_cones)
    m.nlp_load_timer += time() - start

    mip_solution_warmstart = extendMIPSolution(m, mip_solution, new_variable_index_map)

    return old_variable_index_map, mip_solution_warmstart

end



function extendMIPSolution(m::PajaritoConicModel, mip_solution, new_variable_index_map)

    @assert length(mip_solution) == m.numVar

    k = 1
    numVar = m.numVar - m.numIntVar
    new_mip_solution = zeros(numVar)
    for i in 1:m.numVar
        if m.vartype[i] != :Int && m.vartype[i] != :Bin
            new_mip_solution[k] = mip_solution[i]
            k += 1
        end
    end 

    k = 1
    new_solution = [new_mip_solution;zeros(2*m.numSpecCones)]
    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC
        # ADD SOC RELAXATION
        # (y, x) \in SOC => y \geq || x ||
        #   => y + s \geq || x ||
        #   => q = y + s, q \geq || x ||
        #   => q = y + s, (q, x) \in SOC, MIN s
            new_ind = new_variable_index_map[ind]
            sum = 0.0
            for i in new_ind[2:length(new_ind)]
                sum += new_mip_solution[i]^2
            end
            new_solution[k+numVar] = sqrt(sum) - new_mip_solution[new_ind[1]]       
            new_solution[k+numVar+m.numSpecCones] = new_solution[k+numVar] + new_mip_solution[new_ind[1]]    
            k += 1
        elseif cone == :SOCRotated
        # ADD ROTATED SOC RELAXATION
        # (y, z, x) \in SOC => 2yz \geq || x ||^2
        #   => 2y + 2s \geq || x ||^2/z
        #   => q = y + s, 2qz \geq || x ||^2
        #   => q = y + s, (q, z, x) \in ROTATED SOC, MIN s
        # TODO FOLLOWING MUST BE TRIPLE VERIFIED
            new_ind = new_variable_index_map[ind]
            sum = 0.0
            for i in new_ind[3:length(new_ind)]
                sum += new_mip_solution[i]^2
            end
            new_solution[k+numVar+m.numSpecCones] = sum / (2.0 * new_mip_solution[new_ind[2]])       
            new_solution[k+numVar] = new_solution[k+numVar+m.numSpecCones] - new_mip_solution[new_ind[1]]    
            k += 1
        elseif cone == :ExpPrimal
        # ADD EXP RELAXATION
        # {(x,y,z)∈ ℝ 3:y>0,ye^(x/y) ≤ z}
        #   => z + s \geq y exp(x/y)
        #   => q = z + s, q \geq y exp(x/y)
        #   => q = z + s, (x,y,q) \in ExpPrimal, MIN s
            new_ind = new_variable_index_map[ind]
            new_solution[k+numVar] = new_mip_solution[new_ind[2]]*exp(new_mip_solution[new_ind[1]]/new_mip_solution[new_ind[2]]) - new_mip_solution[new_ind[3]]
            new_solution[k+numVar+m.numSpecCones] = new_solution[k+numVar] + new_mip_solution[new_ind[3]]
            k += 1  
        end
    end
    return new_solution
end

function loadConicModel(m::PajaritoConicModel, conic_model, mip_solution)
    
    (I, J, V) = findnz(m.A)
    c_new = copy(m.c)
    b_new = copy(m.b)
    conic_var_cones = copy(m.pajarito_var_cones)
    conic_constr_cones = copy(m.pajarito_constr_cones)

    A_new = sparse(I,J,V, m.numConstr, m.numVar)


    (c_new, A_new, b_new, new_var_cones, old_variable_index_map, new_variable_index_map) = 
        preprocessIntegersOut(m, c_new, A_new, b_new, mip_solution, m.vartype, conic_var_cones, m.numVar)

    (new_constr_cones, A_new, b_new) = removeRedundantRows(m, conic_constr_cones, A_new, b_new)
    @assert size(A_new,1) == length(b_new)

    start = time()
    MathProgBase.loadproblem!(conic_model, c_new, A_new, b_new, new_constr_cones, new_var_cones)
    m.nlp_load_timer += time() - start

    return old_variable_index_map, c_new, A_new
end

function loadRelaxedConicModel(m::PajaritoConicModel, conic_model)
   
    start = time() 
    MathProgBase.loadproblem!(conic_model, m.c, m.A, m.b, m.pajarito_constr_cones, m.pajarito_var_cones)
    m.nlp_load_timer += time() - start

end


function loadInitialRelaxedConicModel(m::PajaritoConicModel, conic_model)
   
    start = time() 
    MathProgBase.loadproblem!(conic_model, m.c_ini, m.A_ini, m.b_ini, m.constr_cones_ini, m.var_cones_ini)
    m.nlp_load_timer += time() - start

end


function getSOCNormValue(ind, separator) 
    sum = vecnorm(separator[ind[2:end]], 2)
    return sum
end

function getSOCValue(ind, separator) 
    sum = vecnorm(separator[ind[2:end]], 2) - separator[ind[1]]
    return sum
end

function getSOCRotatedValue(ind, separator) 
    sum = vecnorm(separator[ind[3:end]], 2)^2 - 2.0*separator[ind[1]]*separator[ind[2]]
    return sum
end

function getSOCAggragaterValue(ind_x, ind_y, separator)
    sum = separator[ind_x]^2/separator[ind_y]
    return sum
end

function getExpValue(ind, separator)
    sum = separator[ind[2]] * exp(separator[ind[1]]/separator[ind[2]]) - separator[ind[3]]
    return sum
end

function getDSOC(ind, separator)
    I = Int[]
    V = Float64[]
    push!(I, ind[1])
    push!(V,-(1.0))
    sum = vecnorm(separator[ind[2:end]], 2)
    for i in ind[2:end]
        push!(I, i)
        push!(V,separator[i]/sum)
    end
    return (I,V)
end

function getDSOCRotated(ind, separator)
    I = Int[]
    V = Float64[]
    push!(I, ind[1])
    push!(V,-(2.0))
    sum = vecnorm(separator[ind[2:end]], 2)^2
    push!(I, ind[2])
    push!(J, -(sum)/separator[ind[2]]^2)
    for i in ind[3:end]
        push!(I, i)
        push!(V,2.0*separator[i]/separator[ind[2]])
    end
    return (I,V)
end

function getDSOCAggragater(ind_x, ind_y, separator)
    I = Int[]
    V = Float64[]
    push!(I, ind_x)
    push!(V, 2*separator[ind_x]/separator[ind_y])
    push!(I, ind_y)
    push!(V, -separator[ind_x]^2/separator[ind_y]^2)
    return (I,V)
end


function getDExp(ind, separator)
    I = Int[]
    V = Float64[]
    push!(I, ind[1])
    push!(V,exp(separator[ind[1]]/separator[ind[2]]))
    push!(I, ind[2])
    push!(V,exp(separator[ind[1]]/separator[ind[2]])*(1.0 - separator[ind[1]] / separator[ind[2]]))
    push!(I, ind[3])
    push!(V,-(1.0))
    return (I,V)
end

function addPrimalCuttingPlanes!(m, mip_model, separator, cb, mip_solution)
    k = 1
    max_violation = -1e+5
    initial_query = false
    if mip_solution == zeros(m.numVar)
        initial_query = true
    end
    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC
            if m.disaggregate_soc != _disagg_soc_on
                f = getSOCValue(ind, separator)
                # IF ALL HAVE DIVISION BY ZERO, IT MUST BE FEASIBLE.
                if getSOCNormValue(ind, separator) == 0.0
                    continue
                end 
                (I,V) = getDSOC(ind, separator)
                new_rhs = -f
                for i in 1:length(I)
                    new_rhs += V[i] * separator[I[i]]
                end
                #new_rhs = (abs(new_rhs) < 1e-9 ? 0.0 : new_rhs)
                viol = vecdot(V, mip_solution[I]) - new_rhs
                if viol > max_violation
                    max_violation = viol
                end
                if cb != []
                    @addLazyConstraint(cb, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
                else
                    @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
                end
            else
                for j = 2:length(ind)
                    # \sum{x_i^2} <= y
                    # ADD APPROXIMATION FOR x_i^2 / y <= t_I[i]
                    f = getSOCAggragaterValue(ind[j], ind[1], separator)
                    (I,V) = getDSOCAggragater(ind[j], ind[1], separator)
                    new_rhs = -f
                    for i in 1:length(I)
                        new_rhs += V[i] * separator[I[i]]
                    end
                    #new_rhs = (abs(new_rhs) < 1e-9 ? 0.0 : new_rhs)
                    viol = vecdot(V, mip_solution[I]) - (initial_query ? 0.0 : getValue(m.mip_t[k][j-1])) - new_rhs
                    if viol > max_violation
                        max_violation = viol
                    end
                    if cb != []
                        @addLazyConstraint(cb, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} - m.mip_t[k][j-1] <= new_rhs)     
                    else
                        @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} - m.mip_t[k][j-1] <= new_rhs)     
                    end         
                end
                k += 1
            end
        elseif cone == :SOC
            f = getSOCRotatedValue(ind, separator)
            # IF ALL HAVE DIVISION BY ZERO, IT MUST BE FEASIBLE.
            if separator[ind[2]] == 0.0
                continue
            end 
            (I,V) = getDSOCRotated(ind, separator)
            new_rhs = -f
            for i in 1:length(I)
                new_rhs += V[i] * separator[I[i]]
            end
            #new_rhs = (abs(new_rhs) < 1e-9 ? 0.0 : new_rhs)
            viol = vecdot(V, mip_solution[I]) - new_rhs
            if viol > max_violation
                max_violation = viol
            end
            if cb != []
                @addLazyConstraint(cb, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
            else
                @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
            end

        elseif cone == :ExpPrimal
            f = getExpValue(ind, separator)
            (I,V) = getDExp(ind, separator)
            new_rhs = -f
            for i in 1:length(I)
                new_rhs += V[i] * separator[I[i]]
            end
            #new_rhs = (abs(new_rhs) < 1e-9 ? 0.0 : new_rhs)
            viol = vecdot(V, mip_solution[I]) - new_rhs
            if viol > max_violation
                max_violation = viol
            end
            if cb != []
                @addLazyConstraint(cb, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
            else
                @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
            end
        end
    end

    return max_violation
end


function getDualSeparator(m, conic_dual, old_variable_index_map)
    separator = zeros(m.numVar)
    for i = 1:length(old_variable_index_map)
        @assert m.vartype[old_variable_index_map[i]] != :Int && m.vartype[old_variable_index_map[i]] != :Bin
        separator[old_variable_index_map[i]] = conic_dual[i]
    end
    return separator
end

function addDualCuttingPlanes!(m, mip_model, separator, cb, mip_solution)

    for i in 1:length(separator)
        if abs(separator[i]) < m.dual_cut_zero_tol
            separator[i] = 0.0
        end
    end

    k = 1
    max_violation = -1e+5
    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC || cone == :SOCRotated || cone == :ExpPrimal || cone == :SDP
            for i = ind
                @assert m.vartype[i] != :Int && m.vartype[i] != :Bin
            end
            viol = -vecdot(separator[ind], mip_solution[ind])
            if viol > max_violation
                max_violation = viol
            end
            if cb != []
                @addLazyConstraint(cb, sum{separator[i] * m.mip_x[i], i in ind} >= 0.0)
            else
                @addConstraint(mip_model, sum{separator[i] * m.mip_x[i], i in ind} >= 0.0)
            end
        end
    end

    return max_violation
end



function loadMIPModel(m::PajaritoConicModel, mip_model)
    @defVar(mip_model, m.l[i] <= x[i=1:m.numVar] <= m.u[i])
    t = Array(Vector{Variable},m.numSOCCones)
    k = 1
    for (cone, ind) in m.pajarito_var_cones
        if m.disaggregate_soc == _disagg_soc_on && cone == :SOC
            @defVar(mip_model, 0.0 <= t[k][j=1:m.dimSOCCones[k]] <= Inf)
            @addConstraint(mip_model, sum{t[k][i], i in 1:m.dimSOCCones[k]} - x[ind[1]] <= 0.0)      
            k += 1        
        end
    end
    @setObjective(mip_model, :Min, dot(m.c,x))
    for (cone,ind) in m.pajarito_constr_cones
        if cone == :Zero
            @addConstraint(mip_model, m.A[ind,:]*x .== m.b[ind])
        elseif cone == :NonNeg
            @addConstraint(mip_model, m.A[ind,:]*x .<= m.b[ind])
        elseif cone == :NonPos
            @addConstraint(mip_model, m.A[ind,:]*x .>= m.b[ind])
        else
            error("unrecognized cone $cone")
        end
    end
    numIntVar = 0
    for i in 1:m.numVar
        setCategory(x[i], m.vartype[i])
        if m.vartype[i] == :Int || m.vartype[i] == :Bin
            numIntVar += 1
        end
    end
    m.mip_x = x
    m.mip_t = t
    m.numIntVar = numIntVar
end

function getFirstPhaseConicModelSolution(m::PajaritoConicModel, inf_dcp_model, old_variable_index_map, mip_solution)


    c_new = [zeros(m.numVar);ones(m.numSpecCones);zeros(m.numSpecCones)]

    inf_dcp_solution = MathProgBase.getsolution(inf_dcp_model)
    @assert length(inf_dcp_solution) == length(old_variable_index_map)

    separator = [mip_solution;zeros(2*m.numSpecCones)]
    for i in 1:length(old_variable_index_map)
        separator[old_variable_index_map[i]] = inf_dcp_solution[i]
    end

    inf_conic_objval = dot(c_new, separator)
    return separator, inf_conic_objval

end

function getInitialConicModelSolution(m::PajaritoConicModel, conic_model)

    dual_vector = try 
        MathProgBase.getdual(conic_model)
    catch
        fill(0.0, size(m.A,1))
    end   
 
    conic_dual = m.c - m.A' * (-dual_vector)
    conic_solution = MathProgBase.getsolution(conic_model)

    conic_objval = dot(m.c, conic_solution)
    return conic_solution, conic_objval, conic_dual
end

function getConicModelSolution(m::PajaritoConicModel, conic_model, old_variable_index_map, mip_solution, c_sub, A_sub)

    dual_vector = try 
        MathProgBase.getdual(conic_model)
    catch
        fill(0.0, size(A_sub,1))
    end   
 
    conic_dual = c_sub - A_sub' * (-dual_vector)
    conic_solution = MathProgBase.getsolution(conic_model)
    reduced_conic_objval = MathProgBase.getobjval(conic_model)

    @assert length(conic_solution) == length(old_variable_index_map)
    @assert length(conic_solution) == m.numVar - m.numIntVar

    separator = copy(mip_solution)
    for i in 1:length(old_variable_index_map)
        @assert m.vartype[old_variable_index_map[i]] != :Int && m.vartype[old_variable_index_map[i]] != :Bin 
        separator[old_variable_index_map[i]] = conic_solution[i]
    end

    conic_objval = dot(m.c, separator)
    #@assert abs(conic_objval-reduced_conic_objval) < 1e-4
    return separator, conic_objval, conic_dual
end

function checkInfeasibility(m::PajaritoConicModel, solution)

    #val = m.b - m.A * solution[1:m.numVar]

    max_violation = -1e+5

    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC
            sum = 0.0
            for i in ind[2:length(ind)]
                sum += solution[i]^2
            end
            viol = sqrt(sum) - solution[ind[1]]
            if viol > max_violation
                max_violation = viol
            end
        elseif cone == :SOCRotated
            sum = 0.0
            for i in ind[3:length(ind)]
                sum += solution[i]^2
            end
            viol = sum - 2.0 * solution[ind[1]] * solution[ind[2]]
            if viol > max_violation
                max_violation = viol
            end
        elseif cone == :ExpPrimal
            viol = solution[ind[2]]*exp(solution[ind[1]]/solution[ind[2]]) - solution[ind[3]]
            if viol > max_violation
                max_violation = viol
            end
        end
    end

    return max_violation

end

function compareIntegerSolutions(m::PajaritoConicModel, sol1, sol2)
    int_ind = filter(i->m.vartype[i] == :Int || m.vartype[i] == :Bin, 1:m.numVar)
    return round(sol1[int_ind]) == round(sol2[int_ind])
end

function completeSOCPDisaggregator(m::PajaritoConicModel, solution)

    new_solution = copy(solution)
    if m.disaggregate_soc == _disagg_soc_on
        for (cone,ind) in m.pajarito_var_cones
            if cone == :SOC
                for i in ind[2:end]
                    push!(new_solution, solution[i]^2)
                end
            end
        end
    end

    return new_solution

end

function MathProgBase.optimize!(m::PajaritoConicModel)

    # TO CLASSIFY THE PROBLEM TYPES
    #=out_file = open("output.txt", "a")
    write(out_file, "$(m.instance) $(m.problem_type)\n") 
    close(out_file)
    return=#

    start = time()

    cputime_mip = 0.0
    cputime_conic = 0.0
    cputime_load = 0.0

    m.prep_timer = 0.0
    m.remove_timer = 0.0
    m.nlp_load_timer = 0.0

    mip_model = Model(solver=m.mip_solver)
    loadMIPModel(m, mip_model)

    # solve Conic model for the MIP solution
    ini_conic_model = MathProgBase.ConicModel(m.cont_solver)
    if m.is_conic_solver && !m.force_primal_cuts
        loadRelaxedConicModel(m, ini_conic_model)
    else
        loadInitialRelaxedConicModel(m, ini_conic_model)
    end

    start_conic = time()
    MathProgBase.optimize!(ini_conic_model)
    cputime_conic += time() - start_conic

    ini_conic_status = MathProgBase.status(ini_conic_model)
    if ini_conic_status == :Optimal || ini_conic_status == :Suboptimal
        if m.numIntVar == 0
            m.solution = MathProgBase.getsolution(ini_conic_model)
            m.objval = MathProgBase.getobjval(ini_conic_model)
            m.status = ini_conic_status
            return
        end
        # Add dual cutting planes if the solver is conic
        if m.is_conic_solver && !m.force_primal_cuts
            (conic_solution, conic_objval, conic_dual) = getInitialConicModelSolution(m,ini_conic_model)
            addDualCuttingPlanes!(m, mip_model, conic_dual, [], zeros(m.numVar)) 
        # Add primal cutting planes if the solver is not conic
        else
            separator = MathProgBase.getsolution(ini_conic_model)
            separator = addSlackValues(m, separator)
            @assert length(separator) == m.numVar
            addPrimalCuttingPlanes!(m, mip_model, separator, [], zeros(m.numVar))
        end
    elseif ini_conic_status == :Infeasible
        warn("Initial Conic Relaxation Infeasible.")
        m.status = :Infeasible
        return    
    # TODO Figure out the details for this condition to hold!   
    elseif ini_conic_status == :Unbounded
        warn("Initial Conic Relaxation Unbounded.")
        m.status = :InfeasibleOrUnbounded
        return
    else 
        warn("Conic Solver Failure.")
        m.status = :Error
        return
    end
    ini_conic_objval = MathProgBase.getobjval(ini_conic_model)

    (m.verbose > 0) && println("\nPajarito started...\n")
    (m.verbose > 0) && println("MICONE algorithm $(m.algorithm) is chosen.")
    (m.verbose > 0) && println("MICONE has $(m.numVar) variables, $(m.numConstr) linear constraints, $(m.numSpecCones) nonlinear cones.")
    (m.verbose > 0) && @printf "Initial objective = %13.5f.\n\n" ini_conic_objval

    # Release the inf_conic_model if applicable
    if applicable(MathProgBase.freemodel!,ini_conic_model)
        MathProgBase.freemodel!(ini_conic_model)
    end

    m.status = :UserLimit
    m.objval = Inf
    iter = 0
    optimality_gap = Inf
    mip_objval = -Inf
    prev_mip_solution = zeros(m.numVar)
    cut_added = false

    conic_status = :Infeasible
    conic_primal = zeros(m.numVar)
    
    function coniccallback(cb)
        if cb != []
            mip_objval = -Inf #MathProgBase.cbgetobj(cb)
            mip_solution = MathProgBase.cbgetmipsolution(cb)[1:m.numVar]
        else
            mip_objval = getObjectiveValue(mip_model)
            mip_solution = getValue(m.mip_x)
        end

        # TODO Enable this after extensive testing!
        #int_ind = filter(i->m.vartype[i] == :Int || m.vartype[i] == :Bin, 1:m.numVar)
        #mip_solution[int_ind] = round(mip_solution[int_ind])        

        conic_objval = Inf

        #@assert abs(mip_objval - dot(m.c, mip_solution)) < 1e-4
        (m.verbose > 2) && println("MIP Vartypes: $(m.vartype)")
        (m.verbose > 2) && println("MIP Solution: $mip_solution")

        separator = Any[]
        # MICONE algorithm
        # TODO: add a second phase if dual information is not available
        if m.is_conic_solver && !m.force_primal_cuts

            conic_model = MathProgBase.ConicModel(m.cont_solver)
            start_load = time()
            (old_variable_index_map, c_sub, A_sub) = loadConicModel(m, conic_model, mip_solution)
            cputime_load += time() - start_load

            # Following is redundant for a Conic solver until warmstart is implemented
            #=if applicable(MathProgBase.setwarmstart!,conic_model, Float64[])
                conic_model_warmstart = Float64[]
                for i in 1:m.numVar
                    if m.vartype[i] != :Int && m.vartype[i] != :Bin
                        push!(conic_model_warmstart, mip_solution[i])
                    end
                end
                MathProgBase.setwarmstart!(conic_model, conic_model_warmstart)
            end=#

            start_conic = time()
            MathProgBase.optimize!(conic_model)
            cputime_conic += time() - start_conic
            conic_status = MathProgBase.status(conic_model)
            (m.verbose > 1) && println("Conic Status: $conic_status from conic solver.")
            if !(conic_status == :Optimal || conic_status == :Suboptimal || conic_status == :Infeasible)
                (m.verbose > 1) && println("ERROR: Unrecognized status $conic_status from conic solver.")
                m.status = :Error
                return
            end

            (conic_primal, conic_objval, conic_dual) = getConicModelSolution(m,conic_model, old_variable_index_map, mip_solution, c_sub, A_sub)

            # Release the conic_model if applicable
            if applicable(MathProgBase.freemodel!,conic_model)
                MathProgBase.freemodel!(conic_model)
            end

            # KEEP TRACK OF BEST KNOWN INTEGER FEASIBLE SOLUTION
            if conic_objval < m.objval
                m.objval = conic_objval
                m.solution = conic_primal[1:m.numVar]
            end 

            # Update separator to dual solution
            separator = getDualSeparator(m, conic_dual, old_variable_index_map)
        # MIDCP algorithm
        else
            conic_status = :Infeasible
            # PHASE 1 INFEASIBILITY PROBLEM
            inf_dcp_model = MathProgBase.ConicModel(m.cont_solver)
            start_load = time()
            (old_variable_index_map, mip_solution_warmstart) = loadFirstPhaseConicModel(m,inf_dcp_model, mip_solution)
            cputime_load += time() - start_load

            if applicable(MathProgBase.setwarmstart!,inf_dcp_model, mip_solution_warmstart)
                MathProgBase.setwarmstart!(inf_dcp_model, mip_solution_warmstart)
            end
            start_conic = time()
            MathProgBase.optimize!(inf_dcp_model)
            cputime_conic += time() - start_conic

            inf_dcp_status = MathProgBase.status(inf_dcp_model)
            if inf_dcp_status != :Optimal && inf_dcp_status != :Suboptimal
                warn("Conic Solver Failure.")
                m.status = :Error
                return
            else
                (separator, inf_conic_objval) = getFirstPhaseConicModelSolution(m,inf_dcp_model, old_variable_index_map, mip_solution)

                # Release the inf_dcp_model if applicable
                if applicable(MathProgBase.freemodel!,inf_dcp_model)
                    MathProgBase.freemodel!(inf_dcp_model)
                end

                if inf_conic_objval > 1e-4
                    (m.verbose > 1) && println("INF DCP Objval: $inf_conic_objval")
                else
                    # PHASE 2 REDUCED PROBLEM
                    dcp_model = MathProgBase.ConicModel(m.cont_solver)

                    start_load = time()
                    (old_variable_index_map, c_sub, A_sub) = loadConicModel(m, dcp_model, mip_solution)
                    cputime_load += time() - start_load

                    if applicable(MathProgBase.setwarmstart!,dcp_model, Float64[])
                        dcp_model_warmstart = Float64[]
                        for i in 1:m.numVar
                            if m.vartype[i] != :Int && m.vartype[i] != :Bin
                                push!(dcp_model_warmstart, separator[i])
                            end
                        end
                        MathProgBase.setwarmstart!(dcp_model, dcp_model_warmstart[1:(m.numVar-m.numIntVar)])
                    end
                    start_conic = time()
                    MathProgBase.optimize!(dcp_model)
                    cputime_conic += time() - start_conic
                    conic_status = MathProgBase.status(dcp_model)
                    (m.verbose > 1) && println("DCP Status: $conic_status from conic solver.")
                    if !(conic_status == :Optimal || conic_status == :Suboptimal)
                        (m.verbose > 1) && println("ERROR: Unrecognized status $conic_status from conic solver.")
                        m.status = :Error
                        return
                    end

                    (conic_primal, conic_objval, dcp_dual) = getConicModelSolution(m,dcp_model, old_variable_index_map, mip_solution, c_sub, A_sub)

                    # Release the dcp_model if applicable
                    if applicable(MathProgBase.freemodel!,dcp_model)
                        MathProgBase.freemodel!(dcp_model)
                    end

                    # KEEP TRACK OF BEST KNOWN INTEGER FEASIBLE SOLUTION
                    if conic_objval < m.objval
                        m.objval = conic_objval
                        m.solution = conic_primal[1:m.numVar]
                    end 
                    
                    # Update separator to primal solution
                    separator = conic_primal
                end
            end
     
        end

        # add supporting hyperplanes
        optimality_gap = m.objval - mip_objval
        primal_infeasibility = checkInfeasibility(m, mip_solution)
        OA_infeasibility = 0.0 
        # ITS FINE TO CHECK OPTIMALITY GAP ONLY BECAUSE IF conic_model IS INFEASIBLE, ITS OBJ VALUE IS INF
        cycle_indicator = (m.algorithm == "OA" ? compareIntegerSolutions(m, prev_mip_solution, mip_solution) : false)
        if (optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance && !cycle_indicator) || cb != [] #&& !(prev_mip_solution == mip_solution)
            if m.is_conic_solver && !m.force_primal_cuts
                OA_infeasibility = addDualCuttingPlanes!(m, mip_model, separator, cb, mip_solution)
            else
                OA_infeasibility = addPrimalCuttingPlanes!(m, mip_model, separator, cb, mip_solution)
            end
            #setValue(m.mip_x, mip_solution)
            cut_added = true
        else
            if optimality_gap < (abs(mip_objval) + 1e-5)*m.opt_tolerance
                (m.verbose > 1) && println("MINLP Solved")
                m.status = :Optimal
                (m.verbose > 1) && println("CPUTIME: $(time() - start)")
                (m.verbose > 1) && println("Number of OA iterations: $iter")
            else #cycle_indicator
                @assert cycle_indicator
                m.status = :Suboptimal
            end
            m.iterations = iter
            #break
        end
        (m.verbose > 0) && (m.algorithm == "OA") && OAprintLevel(iter, mip_objval, conic_objval, optimality_gap, m.objval, primal_infeasibility, OA_infeasibility)
        (cycle_indicator && m.status != :Optimal) && warn("Mixed-integer cycling detected, terminating Pajarito...")
    end

    function heuristiccallback(cb)
        if conic_status == :Optimal
            for i = 1:m.numVar
                setSolutionValue!(cb, m.mip_x[i], conic_primal[i])
            end
            if m.disaggregate_soc == _disagg_soc_on
                k = 1
                for (cone,ind) in m.pajarito_var_cones
                    if cone == :SOC
                        j = 1
                        for i in ind[2:end]
                            setSolutionValue!(cb, m.mip_t[k][j], conic_primal[i]^2)
                            j += 1
                        end
                        k += 1
                    end
                end
            end
            addSolution(cb)
        end
    end

    # BC
    if m.algorithm == "BC"
        addLazyCallback(mip_model,coniccallback)
        addHeuristicCallback(mip_model, heuristiccallback)
        m.status = solve(mip_model)
    # OA
    elseif m.algorithm == "OA"
        (m.verbose > 0) && println("Iteration   MIP Objective   Conic Objective   Optimality Gap   Best Solution    Primal Inf.      OA Inf.")
        while (time() - start) < m.time_limit
            flush(STDOUT)
            cut_added = false
            # gc()
            # WARMSTART MIP FROM UPPER BOUND
            if !any(isnan,m.solution) && !isempty(m.solution)
                warmstart_solution = completeSOCPDisaggregator(m, m.solution) 
                if applicable(MathProgBase.setwarmstart!, getInternalModel(mip_model), warmstart_solution)
                    MathProgBase.setwarmstart!(getInternalModel(mip_model), warmstart_solution)
                end
            end
            # solve MIP model
            start_mip = time()
            mip_status = solve(mip_model)
            cputime_mip += time() - start_mip
            if mip_status == :Infeasible || mip_status == :InfeasibleOrUnbounded
                (m.verbose > 1) && println("MIP Infeasible")
                m.status = :Infeasible
                return
            else 
                (m.verbose > 1) && println("MIP Status: $mip_status")
            end
            mip_solution = getValue(m.mip_x)
            conic_objval = Inf
            coniccallback([])
            if cut_added == false
                break
            end

            prev_mip_solution = mip_solution
            iter += 1
        end

    else
        error("Unspecified algorithm.")
    end

    if m.instance != ""        
        out_file = open("output.txt", "a")
        write(out_file, "$(m.instance): $(m.status) $iter $(time() - start) $(m.objval) $(m.problem_type) $(cputime_mip) $(cputime_conic)\n") 
        close(out_file)
    end

    (m.verbose > 0) && println("\nPajarito finished...\n")
    (m.verbose > 0) && @printf "Status            = %13s.\n" m.status
    (m.verbose > 0) && (m.status == :Optimal) && @printf "Optimum objective = %13.5f.\n" m.objval
    (m.verbose > 0) && (m.algorithm == "OA") && @printf "Iterations        = %13d.\n" iter
    (m.verbose > 0) && @printf "Total time        = %13.5f sec.\n" (time()-start)
    (m.verbose > 0) && @printf "MIP total time    = %13.5f sec.\n" cputime_mip
    (m.verbose > 0) && @printf "CONE total time   = %13.5f sec.\n\n" cputime_conic
    
    (m.profile) && @printf "Profiler:\n"
    (m.profile) && @printf "Preparing Conic subproblem   = %13.5f sec.\n" cputime_load
    (m.profile) && @printf " - Preprocess out integers   = %13.5f sec.\n" m.prep_timer
    (m.profile) && @printf " - Redundant row elimination = %13.5f sec.\n\n" m.remove_timer
    (m.profile) && @printf " - Subproblem load time      = %13.5f sec.\n" m.nlp_load_timer


end


MathProgBase.setwarmstart!(m::PajaritoConicModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PajaritoConicModel, v::Vector{Symbol}) = (m.vartype[1:length(v)] = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.numconstr(m::PajaritoConicModel) = m.numConstr
MathProgBase.numvar(m::PajaritoConicModel) = m.numVar
MathProgBase.status(m::PajaritoConicModel) = m.status
MathProgBase.getobjval(m::PajaritoConicModel) = m.objval
MathProgBase.getsolution(m::PajaritoConicModel) = m.solution
