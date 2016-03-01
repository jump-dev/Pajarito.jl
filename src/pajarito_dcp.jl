using JuMP
using KNITRO
import ECOS

type PajaritoDCPModel <: MathProgBase.AbstractConicModel
    solution::Vector{Float64}
    status
    objval::Float64
    iterations::Int
    numVar::Int
    numVar_ini::Int
    numIntVar::Int
    numConstr::Int
    verbose::Int
    mip_solver
    dcp_solver
    opt_tolerance
    acceptable_opt_tolerance
    time_limit
    cut_switch
    c
    A
    b
    c_ini
    A_ini
    b_ini
    constr_cones_ini
    var_cones_ini
    mip_x
    mip_t
    socp_disaggragater::Bool
    instance::AbstractString
    l::Vector{Float64}
    u::Vector{Float64}
    vartype::Vector{Symbol}
    objsense::Symbol
    numSpecCones
    lengthSpecCones
    numSOCCones
    dimSOCCones
    pajarito_var_cones
    pajarito_constr_cones
    problem_type
    function PajaritoDCPModel(verbose,mip_solver,dcp_solver,opt_tolerance,acceptable_opt_tolerance,time_limit,cut_switch,socp_disaggragater,instance)
        m = new()
        m.verbose = verbose
        m.mip_solver = mip_solver
        m.dcp_solver = dcp_solver
        m.opt_tolerance = opt_tolerance
        m.acceptable_opt_tolerance = acceptable_opt_tolerance
        m.time_limit = time_limit
        m.cut_switch = cut_switch
        m.socp_disaggragater = socp_disaggragater
        m.instance = instance
        return m
    end
end


export PajaritoDCPSolver
immutable PajaritoDCPSolver <: MathProgBase.AbstractMathProgSolver
    verbose
    mip_solver
    dcp_solver
    opt_tolerance
    acceptable_opt_tolerance
    time_limit
    cut_switch
    socp_disaggragater
    instance
end
PajaritoDCPSolver(;verbose=0,mip_solver=CplexSolver(CPX_PARAM_SCRIND=0,CPX_PARAM_REDUCE=0,CPX_PARAM_EPINT=1e-8,CPX_PARAM_EPRHS=1e-8),dcp_solver=ECOS.ECOSSolver(maxit=10000),opt_tolerance=1e-5,acceptable_opt_tolerance=1e-4,time_limit=60*60*10,cut_switch=1,socp_disaggragater=false,instance="") = PajaritoDCPSolver(verbose,mip_solver,dcp_solver,opt_tolerance,acceptable_opt_tolerance,time_limit,cut_switch,socp_disaggragater,instance)

# BEGIN MATHPROGBASE INTERFACE

MathProgBase.ConicModel(s::PajaritoDCPSolver) = PajaritoDCPModel(s.verbose, s.mip_solver, s.dcp_solver, s.opt_tolerance, s.acceptable_opt_tolerance, s.time_limit, s.cut_switch, s.socp_disaggragater, s.instance)

function MathProgBase.loadproblem!(
    m::PajaritoDCPModel, c, A, b, constr_cones, var_cones)

    m.c_ini = c
    m.A_ini = A
    m.b_ini = b
    m.constr_cones_ini = constr_cones
    m.var_cones_ini = var_cones

    numVar = length(c) # number of variables
    numConstr = length(b) # number of constraints

    # @show numConstr, numVar

    # b - Ax \in K => b - Ax = s, s \in K
    pajarito_var_cones = Any[x for x in var_cones]
    pajarito_constr_cones = Any[]
    copy_constr_cones = copy(constr_cones)
    lengthSpecCones = 0
    # ADD SLACKS FOR ONLY SOC AND EXP
    A_I, A_J, A_V = findnz(A)
    slack_count = numVar+1
    for (cone, ind) in copy_constr_cones
        if cone == :SOC || cone == :ExpPrimal
            lengthSpecCones += length(ind)
            #@show cone
            #@show ind
            slack_vars = slack_count:(slack_count+length(ind)-1)
            append!(A_I, ind)
            append!(A_J, slack_vars)
            append!(A_V, ones(length(ind)))
            #@show ind_new
            push!(pajarito_var_cones, (cone, slack_vars))
            push!(pajarito_constr_cones, (:Zero, ind))
            slack_count += length(ind)
        else
            push!(pajarito_constr_cones, (cone, ind))
        end
    end
    #@show constr_cones
    #@show pajarito_constr_cones
    A = sparse(A_I,A_J,A_V)
    # ADD SLACKS FOR ALL CONSTRAINTS
    #=A = [A speye(numConstr)]
    for (cone, ind) in copy_constr_cones
        lengthSpecCones += length(ind)
        push!(pajarito_var_cones, (cone, ind+numVar))
        push!(pajarito_constr_cones, (:Zero, ind))
    end=#

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
        if cone == :SOC || cone == :ExpPrimal
            if cone == :SOC
                soc_indicator = true
                l[ind[1]] = 0.0
                numSOCCones += 1
                push!(dimSOCCones,length(ind)-1)
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
end


function addSlackValues(m::PajaritoDCPModel, separator)
    rhs_value = m.b_ini - m.A_ini * separator[1:m.numVar_ini]
    slack_count = 1
    separator_slack = zeros(m.lengthSpecCones)
    for (cone, ind) in m.constr_cones_ini
        if cone == :SOC || cone == :ExpPrimal
            slack_vars = slack_count:(slack_count+length(ind)-1)
            separator_slack[slack_vars] = rhs_value[ind]
            slack_count += length(ind)
        end
    end
    @assert slack_count == m.lengthSpecCones+1
    return [separator[1:m.numVar_ini];separator_slack]
    #return [separator;rhs_value]
end

function preprocessIntegersOut(c_ini, A_ini, b_ini, mip_solution, vartype, var_cones, numVar)

    c_new = copy(c_ini)
    A_new = copy(A_ini)
    b_new = copy(b_ini)

    k = 1
    removableColumnIndicator = [false for i in 1:numVar]
    new_variable_index_map = [-1 for i in 1:numVar]
    old_variable_index_map = Any[]
    for i in 1:numVar
        if vartype[i] == :Int || vartype[i] == :Bin
            removableColumnIndicator[i] = true
        else
            new_variable_index_map[i] = k
            push!(old_variable_index_map, i)
            k += 1
        end   
    end
    new_var_cones = Any[]
    for (cone, ind) in var_cones
        new_ind = Int[]
        for i in ind
            # THIS ASSUMES INTEGER VARIABLES DO NOT APPEAR IN
            # SPEC CONES
            if (cone == :SOC || cone == :ExpPrimal) && (vartype[i] == :Int || vartype[i] == :Bin)
                error("Integer variable x[$i] inside $cone cone")
            end
            if new_variable_index_map[i] != -1
                push!(new_ind, new_variable_index_map[i])
            end
        end
        push!(new_var_cones, (cone, new_ind))
    end
    #=@show mip_solution
    @show full(c_new)
    @show full(A_new)
    @show full(b_new)
    @show removableColumnIndicator=#

    c_new = c_new[!removableColumnIndicator]
    b_new = b_new - A_new[:,removableColumnIndicator]*mip_solution[removableColumnIndicator]
    A_new = A_new[:,!removableColumnIndicator]

    #=@show new_variable_index_map
    @show old_variable_index_map    
    @show full(c_new)
    @show full(A_new)
    @show full(b_new)
    @show var_cones
    @show new_var_cones=#

    return c_new, A_new, b_new, new_var_cones, old_variable_index_map, new_variable_index_map

end

function removeRedundantRows(constr_cones, A_new, b_new)

    @assert size(A_new, 1) == length(b_new)
    (numConstr,numVar) = size(A_new)
    emptyRow = [false for i in 1:numConstr]
    for i = 1:numConstr
        emptyRowInd = true
        for j = 1:numVar
            if abs(A_new[i,j]) >= 1e-5
                emptyRowInd = false
                break
            end
        end
        emptyRow[i] = emptyRowInd
    end

    k = 1
    new_constraint_index_map = [-1 for i in 1:numConstr]
    old_constraint_index_map = Any[]
    for i in 1:numConstr
        if !emptyRow[i]
            new_constraint_index_map[i] = k
            push!(old_constraint_index_map, i)
            k += 1
        end   
    end

    new_constr_cones = Any[]
    for (cone, ind) in constr_cones
        new_ind = Int[]
        for i in ind
            if (cone == :SOC || cone == :ExpPrimal) && (emptyRow[i])
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



    return new_constr_cones, A_new[!emptyRow,:], b_new[!emptyRow]
end

function loadInfeasibleDCPModel(m::PajaritoDCPModel, inf_dcp_model, mip_solution)
    
    # original variables; slack variables; simplification variables
    c_new = [zeros(m.numVar);ones(m.numSpecCones);zeros(m.numSpecCones)]

    #@show length(c_new)

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
        preprocessIntegersOut(c_new, A_new, b_new, extended_mip_solution, vartype, inf_var_cones, infNumVar)

    (new_constr_cones, A_new, b_new) = removeRedundantRows(inf_constr_cones, A_new, b_new)

    #=@show full(A_new)
    @show b_new
    @show c_new
    @show m.pajarito_var_cones
    @show inf_var_cones=#


    MathProgBase.loadproblem!(inf_dcp_model, c_new, A_new, b_new, new_constr_cones, new_var_cones)


    mip_solution_warmstart = extendMIPSolution(m, mip_solution, new_variable_index_map)

    return old_variable_index_map, mip_solution_warmstart

end

function extendMIPSolution(m::PajaritoDCPModel, mip_solution, new_variable_index_map)

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
        elseif cone == :ExpPrimal
        # ADD EXP RELAXATION
        # {(x,y,z)∈ ℝ 3:y>0,ye^(x/y) ≤ z}
        #   => z + s \geq y exp(x/y)
        #   => q = z + s, q \geq y exp(x/y)
        #   => q = z + s, (x,y,q) \in ExpPrimal, MIN s
            #@show length(new_mip_solution), ind
            #@show length(new_solution), k+m.numVar+m.numSpecCones
            new_ind = new_variable_index_map[ind]
            new_solution[k+numVar] = new_mip_solution[new_ind[2]]*exp(new_mip_solution[new_ind[1]]/new_mip_solution[new_ind[2]]) - new_mip_solution[new_ind[3]]
            new_solution[k+numVar+m.numSpecCones] = new_solution[k+numVar] + new_mip_solution[new_ind[3]]
            k += 1  
        end
    end
    return new_solution
end

function loadDCPModel(m::PajaritoDCPModel, dcp_model, mip_solution)
    
    (I, J, V) = findnz(m.A)
    c_new = copy(m.c)
    b_new = copy(m.b)
    dcp_var_cones = copy(m.pajarito_var_cones)
    dcp_constr_cones = copy(m.pajarito_constr_cones)
    # FIX BINARY OR INTEGER VARIABLES TO CURRENT MIP SOLUTION
    #=k = 1
    for i in 1:m.numVar
        if m.vartype[i] == :Int || m.vartype[i] == :Bin
            push!(I, m.numConstr + k)
            push!(J, i)
            push!(V, 1.0)
            push!(b_new, round(mip_solution[i]))
            k += 1
        end   
    end
    dcp_constr_cones = [dcp_constr_cones;(:Zero, (m.numConstr+1):(length(b_new)))]=#

    A_new = sparse(I,J,V)


    (c_new, A_new, b_new, new_var_cones, old_variable_index_map, new_variable_index_map) = 
        preprocessIntegersOut(c_new, A_new, b_new, mip_solution, m.vartype, dcp_var_cones, m.numVar)

    (new_constr_cones, A_new, b_new) = removeRedundantRows(dcp_constr_cones, A_new, b_new)
    @assert size(A_new,1) == length(b_new)

    MathProgBase.loadproblem!(dcp_model, c_new, A_new, b_new, new_constr_cones, new_var_cones)

    return old_variable_index_map
end

function loadInitialDCPModel(m::PajaritoDCPModel, dcp_model, mip_solution)
    
    (I, J, V) = findnz(m.A_ini)
    c_new = copy(m.c_ini)
    b_new = copy(m.b)
    dcp_var_cones = copy(m.var_cones_ini)
    dcp_constr_cones = copy(m.constr_cones_ini)
    # FIX BINARY OR INTEGER VARIABLES TO CURRENT MIP SOLUTION
    #=k = 1
    for i in 1:m.numVar_ini
        if m.vartype[i] == :Int || m.vartype[i] == :Bin
            push!(I, m.numConstr + k)
            push!(J, i)
            push!(V, 1.0)
            push!(b_new, round(mip_solution[i]))
            k += 1
        end   
    end
    @assert length(b_new) - m.numConstr == m.numIntVar
    dcp_constr_cones = [dcp_constr_cones;(:Zero, (m.numConstr+1):(length(b_new)))]
=#
    A_new = sparse(I,J,V)
    #@show full(A_new)
    #@show b_new
    #@show dcp_constr_cones

    (c_new, A_new, b_new, new_var_cones, old_variable_index_map) = 
        preprocessIntegersOut(c_new, A_new, b_new, mip_solution[1:m.numVar_ini], m.vartype[1:m.numVar_ini], dcp_var_cones, m.numVar_ini)
 

    MathProgBase.loadproblem!(dcp_model, c_new, A_new, b_new, dcp_constr_cones, new_var_cones)

    return old_variable_index_map
end

function loadRelaxedDCPModel(m::PajaritoDCPModel, dcp_model)
    
    MathProgBase.loadproblem!(dcp_model, m.c, m.A, m.b, m.pajarito_constr_cones, m.pajarito_var_cones)

end


function loadInitialRelaxedDCPModel(m::PajaritoDCPModel, dcp_model)
    
    MathProgBase.loadproblem!(dcp_model, m.c_ini, m.A_ini, m.b_ini, m.constr_cones_ini, m.var_cones_ini)

end


function getSOCNormValue(ind, separator) 
    sum = vecnorm(separator[ind[2:end]], 2)
    return sum
end

function getSOCValue(ind, separator) 
    sum = vecnorm(separator[ind[2:end]], 2) - separator[ind[1]]
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

function checkCuttingPlanes!(m, mip_model, separator, mip_solution)
    k = 1
    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC
            if !m.socp_disaggragater
                f = getSOCValue(ind, separator)
                (I,V) = getDSOC(ind, separator) 
                new_rhs = -f
                for i in 1:length(I)
                    new_rhs += V[i] * separator[I[i]]
                end 
                #@show sum(V .* mip_solution[I]) - new_rhs
                #@assert sum(V .* mip_solution[I]) >= new_rhs + 1e-5
                #MathProgBase.addconstr!(mip_model, I, V, -Inf, new_rhs)
            else
                #=for j = 2:length(ind)
                    # \sum{x_i^2} <= y
                    # ADD APPROXIMATION FOR x_i^2 / y <= t_I[i]
                    f = getSOCAggragaterValue(ind[j], ind[1], separator)
                    (I,V) = getDSOCAggragater(ind[j], ind[1], separator)
                    new_rhs = -f
                    for i in 1:length(I)
                        new_rhs += V[i] * separator[I[i]]
                    end
                    @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} - m.mip_t[k][j-1] <= new_rhs)              
                end=#
                k += 1
            end
        elseif cone == :ExpPrimal
            f = getExpValue(ind, separator)
            (I,V) = getDExp(ind, separator)
            new_rhs = -f
            for i in 1:length(I)
                new_rhs += V[i] * separator[I[i]]
            end
            #new_rhs = (abs(new_rhs) < 1e-4 ? 0.0 : new_rhs)
            #@show maximum(abs(V)), minimum(abs(V)), new_rhs
            #@show sum(V .* mip_solution[I]) - new_rhs
            #@assert sum(V .* mip_solution[I]) >= new_rhs + 1e-5
            #MathProgBase.addconstr!(mip_model, I, V, -Inf, new_rhs)
            
        end
    end
end

function addCuttingPlanes!(m, mip_model, separator)
    k = 1
    for (cone, ind) in m.pajarito_var_cones
        if cone == :SOC
            if !m.socp_disaggragater
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
                @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
                #MathProgBase.addconstr!(mip_model, I, V, -Inf, new_rhs)
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
                    @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} - m.mip_t[k][j-1] <= new_rhs)              
                end
                k += 1
            end
        elseif cone == :ExpPrimal
            f = getExpValue(ind, separator)
            (I,V) = getDExp(ind, separator)
            #@show separator[ind], I, V
            new_rhs = -f
            for i in 1:length(I)
                new_rhs += V[i] * separator[I[i]]
            end
            #new_rhs = (abs(new_rhs) < 1e-9 ? 0.0 : new_rhs)
            #@show maximum(abs(V)), minimum(abs(V)), new_rhs
            @addConstraint(mip_model, sum{V[i] * m.mip_x[I[i]], i in 1:length(I)} <= new_rhs)
            #MathProgBase.addconstr!(mip_model, I, V, -Inf, new_rhs)
        end
    end
end

function loadMIPModel(m::PajaritoDCPModel, mip_model)
    @defVar(mip_model, m.l[i] <= x[i=1:m.numVar] <= m.u[i])
    t = Array(Vector{Variable},m.numSOCCones)
    k = 1
    for (cone, ind) in m.pajarito_var_cones
        if m.socp_disaggragater && cone == :SOC
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
    for i in 1:m.numVar
        setCategory(x[i], m.vartype[i])
    end
    m.mip_x = x
    m.mip_t = t
end

function getInfeasibleDCPModelSolution(m::PajaritoDCPModel, inf_dcp_model, old_variable_index_map, mip_solution)


    c_new = [zeros(m.numVar);ones(m.numSpecCones);zeros(m.numSpecCones)]

    inf_dcp_solution = MathProgBase.getsolution(inf_dcp_model)
    @assert length(inf_dcp_solution) == length(old_variable_index_map)

    separator = [mip_solution;zeros(2*m.numSpecCones)]
    for i in 1:length(old_variable_index_map)
        separator[old_variable_index_map[i]] = inf_dcp_solution[i]
    end

    inf_dcp_objval = dot(c_new, separator)
    #@show inf_dcp_objval
    return separator, inf_dcp_objval

end

function getDCPModelSolution(m::PajaritoDCPModel, dcp_model, old_variable_index_map, mip_solution)

    dcp_solution = MathProgBase.getsolution(dcp_model)
    reduced_dcp_objval = MathProgBase.getobjval(dcp_model)

    @assert length(dcp_solution) == length(old_variable_index_map)
    @assert length(dcp_solution) == m.numVar - m.numIntVar

    separator = mip_solution
    for i in 1:length(old_variable_index_map)
        @assert m.vartype[old_variable_index_map[i]] != :Int && m.vartype[old_variable_index_map[i]] != :Bin 
        separator[old_variable_index_map[i]] = dcp_solution[i]
    end

    dcp_objval = dot(m.c, separator)
    @assert abs(dcp_objval-reduced_dcp_objval) < 1e-4
    return separator, dcp_objval
end

function checkFeasibility(m::PajaritoDCPModel, separator)

    val = m.b - m.A * separator[1:m.numVar]
    for (cone, ind) in m.pajarito_constr_cones
        if cone == :Zero
            @assert all(-1e-4 .<= val[ind] .<= 1e-4) 
        elseif cone == :NonNeg
            @assert all(val[ind] .>= -1e-4)
        elseif cone == :NonPos
            @assert all(val[ind] .<= 1e-4)
        end
    end

    for (cone, ind) in m.pajarito_var_cones
        if cone == :NonNeg
            @assert all(separator[ind] .>= -1e-5)
        elseif cone == :NonPos
            @assert all(separator[ind] .<= 1e-5)
        elseif cone == :SOC
            sum = 0.0
            for i in ind[2:length(ind)]
                sum += separator[i]^2
            end
            @assert sqrt(sum) - separator[ind[1]] <= 1e-5
        elseif cone == :ExpPrimal
            @assert separator[ind[2]]*exp(separator[ind[1]]/separator[ind[2]]) - separator[ind[3]] <= 1e-5
        end
    end

end

function MathProgBase.optimize!(m::PajaritoDCPModel)

    # TO CLASSIFY THE PROBLEM TYPES
    #=out_file = open("output.txt", "a")
    write(out_file, "$(m.instance) $(m.problem_type)\n") 
    close(out_file)
    return=#

    start = time()

    cputime_mip = 0.0
    cputime_dcp = 0.0

    mip_model = Model(solver=m.mip_solver)
    loadMIPModel(m, mip_model)

    # solve DCP model for the MIP solution
    ini_dcp_model = MathProgBase.ConicModel(m.dcp_solver)
    loadInitialRelaxedDCPModel(m, ini_dcp_model)
    MathProgBase.optimize!(ini_dcp_model)


    ini_dcp_status = MathProgBase.status(ini_dcp_model)
    ini_dcp_objval = MathProgBase.getobjval(ini_dcp_model)
    (m.verbose > 0) && println("INI DCP STATUS: $ini_dcp_status")
    (m.verbose > 0) && println("INI DCP OBJVAL: $ini_dcp_objval")
    if ini_dcp_status == :Optimal || ini_dcp_status == :Suboptimal
        #dual_solution = MathProgBase.getconicdual(ini_dcp_model)
        #@assert abs(ini_dcp_objval + dot(dual_solution, m.b_ini)) < 1e-4
        #@show ini_dcp_objval, dot(dual_solution, m.b_ini)
        separator = MathProgBase.getsolution(ini_dcp_model)
        separator = addSlackValues(m, separator)
        @assert length(separator) == m.numVar
        addCuttingPlanes!(m, mip_model, separator)
    else
        (m.verbose > 0) && println("DCP Relaxation Infeasible")
        m.status = :Infeasible
        return       
    end

    #@show string(typeof(m.dcp_solver))
    #@show string(typeof(m.dcp_solver.nlp_solver))
    #@show string(typeof(ini_dcp_model.nlp_model))

    if string(typeof(m.dcp_solver)) == "Pajarito.ConicNLPSolver" 
	if string(typeof(m.dcp_solver.nlp_solver)) == "KNITRO.KnitroSolver"
		println("RELEASE KNITRO LICESE")
		KNITRO.freeProblem(getInternalModel(ini_dcp_model.nlp_model).inner)
	end
    end

    # FOR DCP MODEL
    # dcp_model = MathProgBase.model(m.dcp_solver)
    # loadDCPModel(m, dcp_model, mip_solution)

    # FOR INC DCP MODEL
    # inf_dcp_model = MathProgBase.model(m.dcp_solver)
    # loadInfeasibleDCPModel(m,inf_dcp_model, mip_solution)

    #@show m.vartype
    #@show m.pajarito_var_cones

    m.objval = Inf
    iter = 0
    optimality_gap = Inf
    mip_objval = Inf
    prev_mip_solution = zeros(m.numVar)
    while (time() - start) < m.time_limit
	#gc()
        # solve MIP model

	# WARMSTART MIP FROM UPPER BOUND
	if m.objval != Inf
                MathProgBase.setwarmstart!(getInternalModel(mip_model), m.solution)
	end

        start_mip = time()
        mip_status = solve(mip_model)
        cputime_mip += time() - start_mip
        #mip_objval = Inf
        #mip_solution = zeros(m.numVar+1)
        if mip_status == :Infeasible || mip_status == :InfeasibleOrUnbounded
            (m.verbose > 0) && println("MIP Infeasible")
            m.status = :Infeasible
            return
        else 
            (m.verbose > 0) && println("MIP Status: $mip_status")
        end
        mip_objval = getObjectiveValue(mip_model)
        mip_solution = getValue(m.mip_x)
        @assert abs(mip_objval - dot(m.c, mip_solution)) < 1e-4
        (m.verbose > 1) && println("MIP Vartypes: $(m.vartype)")
        (m.verbose > 1) && println("MIP Solution: $mip_solution")

        # PHASE 1 INFEASIBILITY PROBLEM
        dcp_objval = Inf
        inf_dcp_model = MathProgBase.ConicModel(m.dcp_solver)
        (old_variable_index_map, mip_solution_warmstart) = loadInfeasibleDCPModel(m,inf_dcp_model, mip_solution)
        #(string(typeof(m.dcp_solver)) != "ECOSSolver") && MathProgBase.setwarmstart!(inf_dcp_model, mip_solution_warmstart)
        #loadInfeasibleDCPModel(m,inf_dcp_model, mip_solution)
        start_dcp = time()
        MathProgBase.optimize!(inf_dcp_model)
        cputime_dcp += time() - start_dcp

        inf_cut_generator = true
        if MathProgBase.status(inf_dcp_model) == :Infeasible
            (m.verbose > 0) && println("INF DCP Infeasible")
            m.status = :Infeasible
            return
        else
            (separator, inf_dcp_objval) = getInfeasibleDCPModelSolution(m,inf_dcp_model, old_variable_index_map, mip_solution)


	    if string(typeof(m.dcp_solver)) == "Pajarito.ConicNLPSolver" 
		if string(typeof(m.dcp_solver.nlp_solver)) == "KNITRO.KnitroSolver"
			println("RELEASE KNITRO LICESE")
			KNITRO.freeProblem(getInternalModel(inf_dcp_model.nlp_model).inner)
		end
	    end


            #inf_dcp_objval = MathProgBase.getobjval(inf_dcp_model)
            #separator = MathProgBase.getsolution(inf_dcp_model)
            #checkFeasibility(m, separator)
            #@show separator[1:m.numVar]
            #dcp_objval = dot(m.c, separator[1:m.numVar])
            if inf_dcp_objval > 1e-4
                (m.verbose > 0) && println("INF DCP Objval: $inf_dcp_objval")
                inf_cut_generator = true
            else
                #dcp_model_warmstart = MathProgBase.getsolution(inf_dcp_model)
                #@assert all(-1e-5 .<= dcp_model_warmstart[m.numVar+1-m.numIntVar:m.numVar+m.numSpecCones-m.numIntVar] .<= 1e-5)
                dcp_model_warmstart = Float64[]
                for i in 1:m.numVar
                    if m.vartype[i] != :Int && m.vartype[i] != :Bin
                        push!(dcp_model_warmstart, separator[i])
                    end
                end

                #@show separator
                #@show dcp_model_warmstart

                dcp_model = MathProgBase.ConicModel(m.dcp_solver)
                old_variable_index_map = loadDCPModel(m, dcp_model, mip_solution)

                #(string(typeof(m.dcp_solver)) != "ECOSSolver") && MathProgBase.setwarmstart!(dcp_model, dcp_model_warmstart[1:(m.numVar-m.numIntVar)])
                #@show dcp_model.solution
                start_dcp = time()
                MathProgBase.optimize!(dcp_model)
                cputime_dcp += time() - start_dcp
                dcp_status = MathProgBase.status(dcp_model)
                (m.verbose > 0) && println("DCP Status: $dcp_status from conic solver.")
                if !(dcp_status == :Optimal || dcp_status == :Suboptimal)
                    (m.verbose > 0) && println("ERROR: Unrecognized status $dcp_status from conic solver.")
                    m.status = :Error
                    return
                end

                (separator, dcp_objval) = getDCPModelSolution(m,dcp_model, old_variable_index_map, mip_solution)


		if string(typeof(m.dcp_solver)) == "Pajarito.ConicNLPSolver" 
		    if string(typeof(m.dcp_solver.nlp_solver)) == "KNITRO.KnitroSolver"
			println("RELEASE KNITRO LICESE")
			KNITRO.freeProblem(getInternalModel(dcp_model.nlp_model).inner)
		    end
		end

                # KEEP TRACK OF BEST KNOWN INTEGER FEASIBLE SOLUTION
                if dcp_objval < m.objval
                    m.objval = dcp_objval
                    m.solution = separator[1:m.numVar]
                end 

                #checkFeasibility(m, separator)

                #dcp_objval = MathProgBase.getobjval(dcp_model)
                #separator = MathProgBase.getsolution(dcp_model)
                #separator = addSlackValues(m, separator)

                #(m.verbose > 0) && println("DCP Objval: $dcp_objval")
                inf_cut_generator = false
            end
        end
        # solve DCP model for the MIP solution
        #=dcp_model = MathProgBase.model(m.dcp_solver)
        loadInitialDCPModel(m, dcp_model, mip_solution)

        # optiimize the DCP problem
        MathProgBase.optimize!(dcp_model)
        dcp_status = MathProgBase.status(dcp_model)
        (m.verbose > 1) && println("DCP Status: $dcp_status")
        dcp_objval = -Inf
        inf_cut_generator = false
        #separator::Vector{Float64}
        if dcp_status == :Optimal || dcp_status == :Suboptimal
            (m.verbose > 0) && println("DCP Solved")
            dcp_objval = MathProgBase.getobjval(dcp_model)
            separator = MathProgBase.getsolution(dcp_model)
            separator = addSlackValues(m, separator)
            (m.verbose > 1) && println("DCP Solution: $separator")
        elseif dcp_status == :Infeasible || dcp_status == :Unbounded # ecos sometimes returns unbounded when the problem is infeasible
            inf_cut_generator = true
            # FOR INF DCP MODEL
            inf_dcp_model = MathProgBase.model(m.dcp_solver)
            loadInfeasibleDCPModel(m,inf_dcp_model, mip_solution)
            (m.verbose > 0) && println("DCP Infeasible")
            MathProgBase.optimize!(inf_dcp_model)
            if MathProgBase.status(inf_dcp_model) == :Infeasible
                (m.verbose > 0) && println("INF DCP Infeasible")
                m.status = :Infeasible
                return
            end
            (m.verbose > 0) && println("INF DCP Solved")
            separator = MathProgBase.getsolution(inf_dcp_model)
            (m.verbose > 1) && println("INF DCP Solution: $separator")
        else
            println("ERROR: Unrecognized status $dcp_status from conic solver.")
            m.status = :Error
            return
        end=#
        # add supporting hyperplanes
        optimality_gap = m.objval - mip_objval 
        (m.verbose > 0) && println("Optimality Gap: $(m.objval) - $(mip_objval) = $(optimality_gap)")
        # ITS FINE TO CHECK OPTIMALITY GAP ONLY BECAUSE IF dcp_model IS INFEASIBLE, ITS OBJ VALUE IS INF

        if optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance && !(prev_mip_solution == mip_solution)
            addCuttingPlanes!(m, mip_model, separator)
            #checkCuttingPlanes!(m, mip_model, separator, mip_solution)
            setValue(m.mip_x, mip_solution)
            #(m.cut_switch > 0) && addCuttingPlanes!(m, mip_model, mip_solution)
        else
            if optimality_gap < (abs(mip_objval) + 1e-5)*m.acceptable_opt_tolerance
                (m.verbose > 0) && println("MINLP Solved")
                m.status = :Optimal
                #optimality_gap > 0.0 ? m.objval = m.objval : m.objval = mip_objval
                #optimality_gap > 0.0 ? m.solution = m.solution : m.solution = mip_solution
                m.iterations = iter
                (m.verbose > 0) && println("CPUTIME: $(time() - start)")
                (m.verbose > 0) && println("Number of OA iterations: $iter")
                out_file = open("output.txt", "a")
                write(out_file, "$(m.instance): $(m.status) $iter $(time() - start) $(m.objval) $(m.problem_type) $(cputime_mip) $(cputime_dcp)\n") 
                close(out_file)
            end
            return
        end
        #MathProgBase.setvarUB!(nlp_model, m.u)
        #MathProgBase.setvarLB!(nlp_model, m.l)
        prev_mip_solution = mip_solution
        iter += 1
    end
    m.status = :UserLimit
    out_file = open("output.txt", "a")
    write(out_file, "$(m.instance): $(m.status) $iter $(time() - start) $(m.objval) $(m.problem_type) $(cputime_mip) $(cputime_dcp)\n") 
    close(out_file)

end


MathProgBase.setwarmstart!(m::PajaritoDCPModel, x) = (m.solution = x)
function MathProgBase.setvartype!(m::PajaritoDCPModel, v::Vector{Symbol}) 
    m.vartype[1:length(v)] = v
    numIntVar = 0
    for vartype in v
        if vartype == :Int || vartype == :Bin
            numIntVar += 1
        end
    end
    m.numIntVar = numIntVar
end
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::PajaritoDCPModel) = m.status
MathProgBase.getobjval(m::PajaritoDCPModel) = m.objval
MathProgBase.getsolution(m::PajaritoDCPModel) = m.solution
