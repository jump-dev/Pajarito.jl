using JuMP

type ConicNLPModel <: MathProgBase.AbstractConicModel
    solution::Vector{Float64}
    status
    objval::Float64
    nlp_solver
    x
    numVar
    numVar_ini
    numConstr
    constr_cones_map
    var_cones_map
    A
    A_ini
    constr_cones_ini
    var_cones_ini
    constr_cones
    var_cones
    b
    nlp_model
    keeprow # rows not removed by presolve
    function ConicNLPModel(nlp_solver)
        m = new()
        m.nlp_solver = nlp_solver
        return m
    end
end 

export ConicNLPSolver
immutable ConicNLPSolver <: MathProgBase.AbstractMathProgSolver
    nlp_solver
end
ConicNLPSolver(;nlp_solver=IpoptSolver()) = ConicNLPSolver(nlp_solver)
MathProgBase.ConicModel(s::ConicNLPSolver) = ConicNLPModel(s.nlp_solver)

function MathProgBase.loadproblem!(
    m::ConicNLPModel, c, A, b, constr_cones, var_cones)

    nlp_model = Model(solver=m.nlp_solver)
    numVar = length(c) # number of variables
    numConstr = length(b) # number of constraints
    m.A_ini = A
    m.constr_cones_ini = constr_cones
    m.var_cones_ini = var_cones
    #@show size(A,1), length(b)
    @assert size(A,1) == length(b)

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
        @assert all(ind .<= size(A,1))
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
    #@show size(A,1), length(b)
    @assert size(A,1) == length(b)
    # ADD SLACKS FOR ALL CONSTRAINTS
    #=A = [A speye(numConstr)]
    for (cone, ind) in copy_constr_cones
        lengthSpecCones += length(ind)
        push!(pajarito_var_cones, (cone, ind+numVar))
        push!(pajarito_constr_cones, (:Zero, ind))
    end=#


    m.numVar = size(A,2)
    m.numVar_ini = numVar
    @assert m.numVar == numVar + lengthSpecCones
    c = [c;zeros(m.numVar-numVar)]

    # LOAD NLP VARIABLES
    @defVar(nlp_model, x[i=1:m.numVar], start = 1)
    @setObjective(nlp_model, :Min, dot(c,x))


    for (cone, ind) in pajarito_var_cones
        if cone == :Zero
            for i in ind
                setLower(x[i], 0.0)
                setUpper(x[i], 0.0)
            end
        elseif cone == :Free
            # do nothing
        elseif cone == :NonNeg
            for i in ind
                setLower(x[i], 0.0)
            end
        elseif cone == :NonPos
            for i in ind
                setUpper(x[i], 0.0)
            end
        elseif cone == :SOC
            @addNLConstraint(nlp_model, sqrt(sum{x[i]^2, i in ind[2:length(ind)]}) - x[ind[1]] <= 0.0)
            setLower(x[ind[1]], 0.0)
        elseif cone == :ExpPrimal
            #@addNLConstraint(nlp_model, exp(x[ind[1]]) - x[ind[3]] <= 0.0)
            @addNLConstraint(nlp_model, x[ind[2]] * exp(x[ind[1]]/x[ind[2]]) - x[ind[3]] <= 0.0)
            setLower(x[ind[2]], 0.0)
            setLower(x[ind[3]], 0.0)
        end
    end

    removableRowIndicator = [false for i in 1:numConstr]
    # *************** PREPROCESS *******************
    constr_cones_map = [:Zero for i in 1:numConstr]
    for (cone, ind) in pajarito_constr_cones
        for j in ind
            constr_cones_map[j] = cone
        end
    end
    var_cones_map = [:Zero for i in 1:m.numVar]
    for (cone, ind) in pajarito_var_cones
        for j in ind
            var_cones_map[j] = cone
        end
    end
    # FIRST PASS TO IDENTIFY SINGLE VARIABLE ROWS
    nonZeroElements = [Any[] for i in 1:numConstr]
    for i in 1:length(A_I)
        push!(nonZeroElements[A_I[i]], (A_J[i], A_V[i]))
    end
    keeprow = Int[]
    removableRowIndicator = [false for i in 1:numConstr]
    #=for i in 1:numConstr
        if length(nonZeroElements[i]) == 1# && (val == 1.0 || val == -1.0)
            (ind, val) = nonZeroElements[i][1]
            #@show full(A[i,:])
            #@show b[i]
            #@show ind, val
            if constr_cones_map[i] == :Zero
                setLower(x[ind], b[i]/val)
                setUpper(x[ind], b[i]/val)
                #println("x[$ind] == $(b[i]/val)")    
            elseif constr_cones_map[i] == :NonNeg
                if val < 0.0
                    setLower(x[ind], b[i]/val)
                    #println("x[$ind] >= $(b[i]/val)")    
                else
                    setUpper(x[ind], b[i]/val)
                    #println("x[$ind] <= $(b[i]/val)")    
                end
            elseif constr_cones_map[i] == :NonPos
                if val < 0.0
                    setUpper(x[ind], b[i]/val)
                    #println("x[$ind] <= $(b[i]/val)")    
                else
                    setLower(x[ind], b[i]/val)
                    #println("x[$ind] >= $(b[i]/val)")    
                end
            end
            removableRowIndicator[i] = true
        else
            push!(keeprow, i)
        end
    end=#
    #@show rowIndicator
    #@show full(A)
    #@show full(A[remRowInd,:])
    # IDENTIFY VARIABLES THAT CAN BE PREPROCESSED OUT
    removableColumnIndicator = [false for i in 1:m.numVar]
    setValues = zeros(m.numVar)
    #=for i in 1:m.numVar
        ub = getUpper(x[i])
        lb = getLower(x[i])
        if ub == lb && var_cones_map[i] != :SOC && var_cones_map[i] != :ExpPrimal
            removableColumnIndicator[i] = true
            setValues[i] = ub
        end
    end=#
    m.numConstr = size(A,1)
    m.keeprow = keeprow
    # *********************************************

    for (cone,ind) in pajarito_constr_cones
        for i in 1:length(ind)
            if !removableRowIndicator[ind[i]]
                if cone == :Zero
                    @addConstraint(nlp_model, A[ind[i],:]*x .== b[ind[i]])
                    #@addConstraint(nlp_model, A[ind[i],!removableColumnIndicator]*x[!removableColumnIndicator] .== b[ind[i]] - A[ind[i],removableColumnIndicator]*setValues[removableColumnIndicator])
                elseif cone == :NonNeg
                    @addConstraint(nlp_model, A[ind[i],:]*x .<= b[ind[i]])
                    #@addConstraint(nlp_model, A[ind[i],!removableColumnIndicator]*x[!removableColumnIndicator] .<= b[ind[i]] - A[ind[i],removableColumnIndicator]*setValues[removableColumnIndicator])
                elseif cone == :NonPos
                    @addConstraint(nlp_model, A[ind[i],:]*x .>= b[ind[i]])
                    #@addConstraint(nlp_model, A[ind[i],!removableColumnIndicator]*x[!removableColumnIndicator] .>= b[ind[i]] - A[ind[i],removableColumnIndicator]*setValues[removableColumnIndicator])
                else
                    error("unrecognized cone $cone")
                end
            end
        end
    end

    @assert size(A,2) == m.numVar
    @assert size(A,1) == length(b)
    @assert length(b) == m.numConstr

    m.x = x
    m.nlp_model = nlp_model
    m.constr_cones_map = constr_cones_map 
    m.A = A
    m.b = b
    m.constr_cones = pajarito_constr_cones
    m.var_cones = pajarito_var_cones
end


function checkFeasibility(m::ConicNLPModel, separator)

    val = m.b - m.A * separator
    for (cone, ind) in m.constr_cones
        if cone == :Zero
            @assert all(-1e-4 .<= val[ind] .<= 1e-4) 
        elseif cone == :NonNeg
            @assert all(val[ind] .>= -1e-4)
        elseif cone == :NonPos
            @assert all(val[ind] .<= 1e-4)
        end
    end

    for (cone, ind) in m.var_cones
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

function MathProgBase.optimize!(m::ConicNLPModel)

    #curr_solution = getValue(m.x)
    #@show curr_solution
    m.status = solve(m.nlp_model)
    m.objval = getObjectiveValue(m.nlp_model)
    
    if (m.status != :Infeasible)
        m.solution = getValue(m.x)
        #checkFeasibility(m, m.solution)    
    end
   
end

function MathProgBase.setwarmstart!(m::ConicNLPModel, x) 

    @assert length(x) == m.numVar_ini
    x_expanded = copy(x)
    val = m.b - m.A_ini*x
    nonlinear_cones = 0
    for (cone, ind) in m.constr_cones_ini
        if cone == :SOC || cone == :ExpPrimal
            append!(x_expanded, val[ind])
            nonlinear_cones += 1
        elseif cone == :NonNeg
            #@assert all(val[ind] .>= -1e-5)
        elseif cone == :NonPos
            #@assert all(val[ind] .<= 1e-5)
        elseif cone == :Zero
            #@assert all(-1e-5 .<= val[ind] .<= 1e-5)
        end
    end
    m.solution = x_expanded
    setValue(m.x, m.solution)
end

MathProgBase.setvartype!(m::ConicNLPModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::ConicNLPModel) = m.status
MathProgBase.getobjval(m::ConicNLPModel) = m.objval
MathProgBase.getsolution(m::ConicNLPModel) = m.solution[1:m.numVar_ini]
