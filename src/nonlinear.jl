#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP

type PajaritoModel <: MathProgBase.AbstractNonlinearModel
    solution::Vector{Float64}
    status
    objval::Float64
    iterations::Int
    numVar::Int
    numConstr::Int
    numNLConstr::Int

    # SOLVER DATA:
    verbose::Int                # Verbosity level flag
    algorithm                   # Choice of algorithm: "OA" or "BC"
    mip_solver                  # Choice of MILP solver
    cont_solver                 # Choice of Conic solver
    opt_tolerance               # Relative optimality tolerance
    time_limit                  # Time limit
    profile::Bool               # Performance profile switch
    disaggregate_soc::Symbol    # SOCP disaggregator for SOC constraints
    instance::AbstractString    # Path to instance
    
    A
    A_lb
    A_ub
    lb::Vector{Float64}
    ub::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    c::Vector{Float64}
    vartype::Vector{Symbol}
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
    objsense::Symbol
    objlinear::Bool
    mip_x
    d

    nlp_load_timer

    # CONSTRUCTOR:
    function PajaritoModel(verbose,algorithm,mip_solver,cont_solver,opt_tolerance,time_limit,profile,disaggregate_soc,instance)
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
        return m
    end
end

type InfeasibleNLPEvaluator <: MathProgBase.AbstractNLPEvaluator
    d
    numConstr::Int
    numNLConstr::Int
    numVar::Int
    constrtype::Vector{Symbol}
    constrlinear::Vector{Bool}
end

function MathProgBase.eval_f(d::InfeasibleNLPEvaluator, x)
    retval = 0.0
    # SUM UP THE SLACKS AND RETURN
    for i in d.numVar+1:length(x)
        retval += x[i]
    end
    return retval
end

function MathProgBase.eval_grad_f(d::InfeasibleNLPEvaluator, g, x)
    g[:] = [zeros(d.numVar); ones(d.numNLConstr)]
end


function MathProgBase.eval_g(d::InfeasibleNLPEvaluator, g, x)
    MathProgBase.eval_g(d.d, g, x[1:d.numVar])
    k = 1
    for i in 1:d.numConstr
        d.constrlinear[i] && continue
        if d.constrtype[i] == :(<=)
            g[i] -= x[k+d.numVar]
        else
            g[i] += x[k+d.numVar]
        end
        k += 1
    end
end

function MathProgBase.jac_structure(d::InfeasibleNLPEvaluator)
    I, J = MathProgBase.jac_structure(d.d)
    I_new = copy(I)
    J_new = copy(J)
    #push indices
    k = 1
    for i in 1:(d.numConstr)
        d.constrlinear[i] && continue
        push!(I_new, i); push!(J_new, k+d.numVar);
        k += 1
    end
    return I_new, J_new
end

function MathProgBase.eval_jac_g(d::InfeasibleNLPEvaluator, J, x) 
    MathProgBase.eval_jac_g(d.d, J, x[1:d.numVar])
    k = length(J) - d.numNLConstr + 1
    for i in 1:d.numConstr
        d.constrlinear[i] && continue
        if d.constrtype[i] == :(<=)
            J[k] = -(1.0)
        else
            J[k] = 1.0
        end
        k += 1
    end
end

function MathProgBase.eval_hesslag(d::InfeasibleNLPEvaluator, H, x, σ, μ)
    MathProgBase.eval_hesslag(d.d, H, x[1:d.numVar], 0.0, μ)
end

MathProgBase.hesslag_structure(d::InfeasibleNLPEvaluator) = MathProgBase.hesslag_structure(d.d)
MathProgBase.initialize(d::InfeasibleNLPEvaluator, requested_features::Vector{Symbol}) = 
MathProgBase.initialize(d.d, requested_features)
MathProgBase.features_available(d::InfeasibleNLPEvaluator) = [:Grad,:Jac,:Hess]
function MathProgBase.eval_jac_prod(d::InfeasibleNLPEvaluator, y, x, w)
    jac_I, jac_J = MathProgBase.jac_structure(d)
    jac_V = zeros(length(jac_I))
    MathProgBase.eval_jac_g(d, jac_V, x)
    varidx_new = [zeros(0) for i in 1:m.numConstr]
    coef_new = [zeros(0) for i in 1:m.numConstr]
    for k in 1:length(jac_I)
        row = jac_I[k]
        push!(varidx_new[row], jac_J[k]); push!(coef_new[row], jac_V[k])
    end

    for i = 1:d.numConstr
        retval = 0.0
        for j in 1:length(varidx_new[i])
            retval += coef_new[i][j] * w[varidx_new[i][j]]
        end
        y[i] = retval
    end
end

#DONT NEED IT FOR NOW
#eval_jac_prod_t(d::AbstractNLPEvaluator, y, x, w)
#eval_hesslag_prod(d::AbstractNLPEvaluator, h, x, v, σ, μ)

MathProgBase.isobjlinear(d::InfeasibleNLPEvaluator) = true
MathProgBase.isobjquadratic(d::InfeasibleNLPEvaluator) = true #MathProgBase.isobjquadratic(d.d)
MathProgBase.isconstrlinear(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.isconstrlinear(d.d, i)
MathProgBase.obj_expr(d::InfeasibleNLPEvaluator) = MathProgBase.obj_expr(d.d)
MathProgBase.constr_expr(d::InfeasibleNLPEvaluator, i::Int) = MathProgBase.constr_expr(d.d, i)

# BEGIN MATHPROGBASE INTERFACE
MathProgBase.NonlinearModel(s::PajaritoSolver) = PajaritoModel(s.verbose, s.algorithm, s.mip_solver, s.cont_solver, s.opt_tolerance, s.time_limit, s.profile, s.disaggregate_soc, s.instance)

function MathProgBase.loadproblem!(
    m::PajaritoModel, numVar, numConstr, l, u, lb, ub, sense, d)

    if !applicable(MathProgBase.NonlinearModel, m.cont_solver)
        error("$(m.cont_solver) is not a nonlinear solver.")
    end

    m.numVar = numVar
    m.numConstr = numConstr
    m.lb = lb
    m.ub = ub
    m.l = l
    m.u = u
    m.objsense = sense
    m.d = d
    m.vartype = fill(:Cont,numVar)

    MathProgBase.initialize(d, [:Grad,:Jac,:Hess])

    m.constrtype = Array(Symbol, numConstr)
    for i = 1:numConstr
        if lb[i] > -Inf && ub[i] < Inf
            m.constrtype[i] = :(==)
        elseif lb[i] > -Inf
            m.constrtype[i] = :(>=)
        else
            m.constrtype[i] = :(<=)
        end
    end
end

function populatelinearmatrix(m::PajaritoModel)
    # set up map of linear rows
    constrlinear = Array(Bool, m.numConstr)
    numlinear = 0
    constraint_to_linear = fill(-1,m.numConstr)
    for i = 1:m.numConstr
        constrlinear[i] = MathProgBase.isconstrlinear(m.d, i)
        if constrlinear[i] 
            numlinear += 1
            constraint_to_linear[i] = numlinear
        end
    end
    m.numNLConstr = m.numConstr - numlinear

    # extract sparse jacobian structure
    jac_I, jac_J = MathProgBase.jac_structure(m.d)

    # evaluate jacobian at x = 0
    c = zeros(m.numVar)
    x = m.solution
    jac_V = zeros(length(jac_I))
    MathProgBase.eval_jac_g(m.d, jac_V, x)
    MathProgBase.eval_grad_f(m.d, c, x)
    m.objlinear = MathProgBase.isobjlinear(m.d)
    if m.objlinear
        (m.verbose > 0) && println("Objective function is linear")
        m.c = c
    else 
        (m.verbose > 0) && println("Objective function is nonlinear")
        m.c = zeros(m.numVar)
    end

    # Build up sparse matrix for linear constraints
    A_I = Int[]
    A_J = Int[]
    A_V = Float64[]

    for k in 1:length(jac_I)
        row = jac_I[k]
        if !constrlinear[row]
            continue
        end
        row = constraint_to_linear[row]
        push!(A_I,row); push!(A_J, jac_J[k]); push!(A_V, jac_V[k])
    end

    m.A = sparse(A_I, A_J, A_V, m.numConstr-m.numNLConstr, m.numVar)

    # g(x) might have a constant, i.e., a'x + b
    # let's find b
    constraint_value = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, constraint_value, x)
    b = constraint_value[constrlinear] - m.A * x
    # so linear constraints are of the form lb ≤ a'x + b ≤ ub

    # set up A_lb and A_ub vectors
    m.A_lb = m.lb[constrlinear] - b
    m.A_ub = m.ub[constrlinear] - b

    # Now we have linear parts
    m.constrlinear = constrlinear



end

function addCuttingPlanes!(m::PajaritoModel, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
    max_violation = -1e+5
    # EVALUATE g and jac_g AT MIP SOLUTION THAT IS INFEASIBLE
    g = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, g, separator[1:m.numVar])
    MathProgBase.eval_jac_g(m.d, jac_V, separator[1:m.numVar])

    # create rows corresponding to constraints in sparse format

    varidx_new = [zeros(Int, 0) for i in 1:m.numConstr]
    coef_new = [zeros(0) for i in 1:m.numConstr]

    for k in 1:length(jac_I)
        row = jac_I[k]
        push!(varidx_new[row], jac_J[k]); push!(coef_new[row], jac_V[k])
    end

    # CREATE CONSTRAINT CUTS
    for i in 1:m.numConstr
        if m.constrtype[i] == :(<=)
            val = g[i] - m.ub[i]
        else
            val = m.lb[i] - g[i]
        end
        lin = m.constrlinear[i]
        (m.verbose > 1) && println("Constraint $i value $val linear $lin")
        if !(lin) #&& (val > 1e-4) # m.ub[i] is in the constraint somehow
            # CREATE SUPPORTING HYPERPLANE
            (m.verbose > 1) && println("Create supporting hyperplane for constraint $i")
            new_rhs::Float64
            if m.constrtype[i] == :(<=)
                new_rhs = m.ub[i] - g[i]
            else
                new_rhs = m.lb[i] - g[i]
            end
            for j = 1:length(varidx_new[i])
                new_rhs += coef_new[i][j] * separator[Int(varidx_new[i][j])]
            end
            (m.verbose > 1) && println("varidx $(varidx_new[i])") 
            (m.verbose > 1) && println("coef $(coef_new[i])") 
            (m.verbose > 1) && println("rhs $new_rhs") 
            if m.constrtype[i] == :(<=)
                if cb != []
                    @addLazyConstraint(cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                else
                    @addConstraint(mip_model, dot(coef_new[i], m.mip_x[varidx_new[i]]) <= new_rhs)
                end
                viol = vecdot(coef_new[i], mip_solution[varidx_new[i]]) - new_rhs
                if viol > max_violation
                    max_violation = viol
                end
                #MathProgBase.addconstr!(mip_model, varidx_new[i], coef_new[i], -Inf, new_rhs)
            else
                if cb != []
                    @addLazyConstraint(cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                else
                    @addConstraint(mip_model, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                end
                viol = new_rhs - vecdot(coef_new[i], mip_solution[varidx_new[i]])
                if viol > max_violation
                    max_violation = viol
                end
                #MathProgBase.addconstr!(mip_model, varidx_new[i], coef_new[i], new_rhs, Inf)
            end 
        end
    end
    # CREATE OBJECTIVE CUTS   
    if !(m.objlinear)
        (m.verbose > 1) && println("Create supporting hyperplane for objective f(x) <= t")
        f = MathProgBase.eval_f(m.d, separator[1:m.numVar])  
        MathProgBase.eval_grad_f(m.d, grad_f, separator[1:m.numVar]) 
        if m.objsense == :Max
            f = -f
            grad_f = -grad_f
        end
        new_rhs = -f
        varidx = zeros(Int, m.numVar+1)
        for j = 1:m.numVar
            varidx[j] = j
            new_rhs += grad_f[j] * separator[j]
        end
        varidx[m.numVar+1] = m.numVar+1
        grad_f[m.numVar+1] = -(1.0)
        (m.verbose > 1) && println("varidx $(varidx)") 
        (m.verbose > 1) && println("coef $(grad_f)") 
        (m.verbose > 1) && println("rhs $new_rhs") 
        if cb != []
            @addLazyConstraint(cb, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        else
            @addConstraint(mip_model, dot(grad_f, m.mip_x[varidx]) <= new_rhs)
        end
        viol = vecdot(grad_f, mip_solution[varidx]) - new_rhs
        if viol > max_violation
            max_violation = viol
        end
        #MathProgBase.addconstr!(mip_model, varidx, grad_f, -Inf, new_rhs)
    end

    return max_violation
end

function loadMIPModel(m::PajaritoModel, mip_model)
    lb = [m.l; -1e6]
    ub = [m.u; 1e6]
    @defVar(mip_model, lb[i] <= x[i=1:m.numVar+1] <= ub[i])
    for i = 1:m.numVar
        setCategory(x[i], m.vartype[i])
    end
    setCategory(x[m.numVar+1], :Cont)
    for i = 1:m.numConstr-m.numNLConstr
        if m.A_lb[i] > -Inf && m.A_ub[i] < Inf
            @addConstraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .>= m.A_lb[i])
            @addConstraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .<= m.A_ub[i])
        elseif m.A_lb[i] > -Inf
            @addConstraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .>= m.A_lb[i])
        else
            @addConstraint(mip_model, m.A[i:i,:]*x[1:m.numVar] .<= m.A_ub[i])
        end
    end
    c_new = [m.objsense == :Max ? -m.c : m.c; m.objlinear ? 0.0 : 1.0]
    @setObjective(mip_model, Min, dot(c_new, x))

    m.mip_x = x
    #=
    mip_model = MathProgBase.LinearQuadraticModel(m.mip_solver)
    MathProgBase.loadproblem!(mip_model,
        [m.A spzeros(size(m.A,1),1)],
        [m.l; -1e4],
        [m.u; Inf],
        [(m.objsense == :Max)? -m.c : m.c; m.objlinear? 0.0 : 1.0], m.A_lb, m.A_ub, :Min)
    MathProgBase.setvartype!(mip_model, [m.vartype; :Cont])
    =#
end

function checkInfeasibility(m::PajaritoModel, solution)

    g = zeros(m.numConstr)
    g_val = -1e+5*ones(m.numConstr)
    MathProgBase.eval_g(m.d, g, solution[1:m.numVar])  
    for i = 1:m.numConstr
        if !m.constrlinear[i]
            if m.constrtype[i] == :(<=)
                g_val[i] = g[i] - m.ub[i]
            else
                g_val[i] = m.lb[i] - g[i]
            end
        end
    end

    return maximum(g_val)

end



function compareIntegerSolutions(m::PajaritoModel, sol1, sol2)
    int_ind = filter(i->m.vartype[i] == :Int || m.vartype[i] == :Bin, 1:m.numVar)
    return round(sol1[int_ind]) == round(sol2[int_ind])
end

function MathProgBase.optimize!(m::PajaritoModel)

    start = time()

    cputime_nlp = 0.0
    cputime_mip = 0.0

    m.nlp_load_timer = 0.0

    populatelinearmatrix(m)
    # solve it
    # MathProgBase.optimize!(nlp_model)

    # pull out solution values
    #= m.status = MathProgBase.status(nlp_model)
    m.objval = MathProgBase.getobjval(nlp_model)
    m.solution = MathProgBase.getsolution(nlp_model) =#

    # MODIFICATION new objective t >= f(x)
    # set up cplex to solve the mixed integer linear problem
    mip_model = Model(solver=m.mip_solver)
    loadMIPModel(m, mip_model)

    for i in 1:m.numConstr
        if !(m.constrlinear[i]) && m.constrtype[i] == :(==)
            error("Nonlinear equality or two-sided constraints not accepted.")
        end
    end

    jac_I, jac_J = MathProgBase.jac_structure(m.d)
    jac_V = zeros(length(jac_I))
    grad_f = zeros(m.numVar+1)

    ini_nlp_model = MathProgBase.NonlinearModel(m.cont_solver)
    start_load = time()
    MathProgBase.loadproblem!(ini_nlp_model,
    m.numVar, m.numConstr, m.l, m.u, m.lb, m.ub, m.objsense, m.d)
    m.nlp_load_timer += time() - start_load

    # pass in starting point
    #MathProgBase.setwarmstart!(nlp_model, m.solution)
    MathProgBase.setwarmstart!(ini_nlp_model, m.solution[1:m.numVar])

    start_nlp = time()
    MathProgBase.optimize!(ini_nlp_model)
    cputime_nlp += time() - start_nlp

    ini_nlp_status = MathProgBase.status(ini_nlp_model)
    if ini_nlp_status == :Optimal || ini_nlp_status == :Suboptimal
        separator = MathProgBase.getsolution(ini_nlp_model)
        addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, [], zeros(m.numVar+1))
    elseif ini_nlp_status == :Infeasible
        warn("Initial NLP Relaxation Infeasible.")
        m.status = :Infeasible
        m.solution = fill(NaN, m.numVar)
        return     
    # TODO Figure out the conditions for this to hold!  
    elseif ini_nlp_status == :Unbounded
        warn("Initial NLP Relaxation Unbounded.")
        m.status = :InfeasibleOrUnbounded
        m.solution = fill(NaN, m.numVar)
        return
    else 
        warn("NLP Solver Failure.")
        m.status = :Error
        m.solution = fill(NaN, m.numVar)
        return
    end
    ini_nlp_objval = MathProgBase.getobjval(ini_nlp_model)

    (m.verbose > 0) && println("\nPajarito started...\n")
    (m.verbose > 0) && println("MINLP algorithm $(m.algorithm) is chosen.")
    (m.verbose > 0) && println("MINLP has $(m.numVar) variables, $(m.numConstr - m.numNLConstr) linear constraints, $(m.numNLConstr) nonlinear constraints.")
    (m.verbose > 0) && @printf "Initial objective = %13.5f.\n\n" ini_nlp_objval

    m.status = :UserLimit
    m.objval = Inf
    iter = 0
    prev_mip_solution = zeros(m.numVar)
    cut_added = false

    nlp_status = :Infeasible
    nlp_solution = zeros(m.numVar)

    function nonlinearcallback(cb)
        if cb != []
            mip_objval = -Inf #MathProgBase.cbgetobj(cb)
            mip_solution = MathProgBase.cbgetmipsolution(cb)[1:m.numVar+1]
        else
            mip_objval = getObjectiveValue(mip_model)
            mip_solution = getValue(m.mip_x)
        end
        #MathProgBase.getsolution(getInternalModel(mip_model))
        (m.verbose > 2) && println("MIP Solution: $mip_solution")
        # solve NLP model for the MIP solution
        new_u = m.u
        new_l = m.l
        for i in 1:m.numVar
            if m.vartype[i] == :Int || m.vartype[i] == :Bin
                new_u[i] = mip_solution[i]
                new_l[i] = mip_solution[i]
            end   
        end

        # set up ipopt to solve continuous relaxation
        nlp_model = MathProgBase.NonlinearModel(m.cont_solver)
        start_load = time()
        MathProgBase.loadproblem!(nlp_model,
        m.numVar, m.numConstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)
        m.nlp_load_timer += time() - start_load

        # pass in starting point
        #MathProgBase.setwarmstart!(nlp_model, m.solution)
        MathProgBase.setwarmstart!(nlp_model, mip_solution[1:m.numVar])

        l_inf = [new_l;zeros(m.numNLConstr)]
        u_inf = [new_u;Inf*ones(m.numNLConstr)]

        d_inf = InfeasibleNLPEvaluator(m.d, m.numConstr, m.numNLConstr, m.numVar, m.constrtype, m.constrlinear)
        inf_model = MathProgBase.NonlinearModel(m.cont_solver)
        start_load = time()
        MathProgBase.loadproblem!(inf_model,
        m.numVar+m.numNLConstr, m.numConstr, l_inf, u_inf, m.lb, m.ub, :Min, d_inf) 
        m.nlp_load_timer += time() - start_load

        #MathProgBase.setvarUB!(nlp_model, new_u)
        #MathProgBase.setvarLB!(nlp_model, new_l)
        # optimize the NLP problem
        start_nlp = time()
        MathProgBase.optimize!(nlp_model)
        cputime_nlp += time() - start_nlp
        nlp_status = MathProgBase.status(nlp_model)
        nlp_objval = -Inf
        #separator::Vector{Float64}
        if nlp_status == :Optimal
            (m.verbose > 2) && println("NLP Solved")
            nlp_objval = MathProgBase.getobjval(nlp_model)
            nlp_solution = MathProgBase.getsolution(nlp_model)
            separator = copy(nlp_solution)
            (m.verbose > 2) && println("NLP Solution: $separator")

            # KEEP TRACK OF BEST KNOWN INTEGER FEASIBLE SOLUTION
            if m.objsense == :Max
                if nlp_objval > -m.objval
                    m.objval = -nlp_objval
                    m.solution = separator[1:m.numVar]
                end
            else
                if nlp_objval < m.objval
                    m.objval = nlp_objval
                    m.solution = separator[1:m.numVar]
                end 
            end

        else
            # Create the warm start solution for inf model
            inf_initial_solution = zeros(m.numVar + m.numNLConstr);
            inf_initial_solution[1:m.numVar] = mip_solution[1:m.numVar] 
            g = zeros(m.numConstr)
            MathProgBase.eval_g(m.d, g, inf_initial_solution[1:m.numVar])
            k = 1
            for i in 1:m.numConstr
                if !m.constrlinear[i]
                    if m.constrtype[i] == :(<=)
                        val = g[i] - m.ub[i]
                    else
                        val = m.lb[i] - g[i]
                    end
                    if val > 0
                        # Because the sign of the slack changes if the constraint
                        # direction change
                        inf_initial_solution[m.numVar + k] = val
                    else
                        inf_initial_solution[m.numVar + k] = 0.0
                    end
                    k += 1
                end
            end

            MathProgBase.setwarmstart!(inf_model, inf_initial_solution)
            (m.verbose > 2) && println("NLP Infeasible")
            start_nlp = time()
            MathProgBase.optimize!(inf_model)
            cputime_nlp += time() - start_nlp
            inf_model_status = MathProgBase.status(inf_model)
            if inf_model_status != :Optimal && inf_model_stauts != :Suboptimal
                warn("NLP Solver Failure.")
                m.status = :Error
                m.solution = fill(NaN, m.numVar)
                return
            end
            (m.verbose > 2) && println("INF NLP Solved")
            separator = MathProgBase.getsolution(inf_model)
            (m.verbose > 2) && println("INF NLP Solution: $separator")
        end
        # add supporting hyperplanes
        cycle_indicator = (m.algorithm == "OA" ? compareIntegerSolutions(m, prev_mip_solution, mip_solution) : false)
        if m.objsense == :Min
            optimality_gap = m.objval - mip_objval
        else
            optimality_gap = -m.objval - mip_objval
        end
        primal_infeasibility = checkInfeasibility(m, mip_solution)
        OA_infeasibility = 0.0
        if (optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance && !cycle_indicator) || cb != []
            OA_infeasibility = addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
            #(m.cut_switch > 0) && addCuttingPlanes!(m, mip_model, mip_solution, jac_I, jac_J, jac_V, grad_f, cb, mip_solution)
            cut_added = true
        else
            if optimality_gap < (abs(mip_objval) + 1e-5)*m.opt_tolerance
                (m.verbose > 1) && println("MINLP Solved")
                m.status = :Optimal
                m.iterations = iter
                (m.verbose > 1) && println("Number of OA iterations: $iter")
            else
                @assert cycle_indicator 
                m.status = :Suboptimal
            end
        end
        (m.verbose > 0) && (m.algorithm == "OA") && @printf "%9d   %+.7e   %+.7e   %+.7e   %+.7e   %+.7e   %+.7e\n" iter mip_objval nlp_objval optimality_gap m.objval primal_infeasibility OA_infeasibility
        (cycle_indicator && m.status != :Optimal) && warn("Mixed-integer cycling detected, terminating Pajarito...")
    end

    function heuristiccallback(cb)
        if nlp_status == :Optimal
            for i = 1:m.numVar
                setSolutionValue!(cb, m.mip_x[i], nlp_solution[i])
            end
            addSolution(cb)
        end
    end

    if m.algorithm == "BC"
        addLazyCallback(mip_model, nonlinearcallback)
        addHeuristicCallback(mip_model, heuristiccallback)
        m.status = solve(mip_model)
    elseif m.algorithm == "OA"
        (m.verbose > 0) && println("Iteration   MIP Objective     NLP Objective   Optimality Gap   Best Solution    Primal Inf.      OA Inf.")
        while (time() - start) < m.time_limit
            flush(STDOUT)
            cut_added = false
            # solve MIP model
            start_mip = time()
            mip_status = solve(mip_model)
            cputime_mip += time() - start_mip
            #mip_objval = Inf
            #mip_solution = zeros(m.numVar+1)
            if mip_status == :Infeasible || mip_status == :InfeasibleOrUnbounded
                (m.verbose > 1) && println("MIP Infeasible")
                m.status = :Infeasible
                m.solution = fill(NaN, m.numVar)
                return
            else 
                (m.verbose > 1) && println("MIP Status: $mip_status")
            end
            mip_solution = getValue(m.mip_x)

            nonlinearcallback([])

            if cut_added == false
                break
            end

            prev_mip_solution = mip_solution
            iter += 1
        end
    else
        error("Unspecified algorithm.")
    end

    if m.objsense == :Max
        m.objval = -m.objval
    end

    (m.verbose > 0) && println("\nPajarito finished...\n")
    (m.verbose > 0) && @printf "Status            = %13s.\n" m.status
    (m.verbose > 0) && (m.status == :Optimal) && @printf "Optimum objective = %13.5f.\n" m.objval
    (m.verbose > 0) && (m.algorithm == "OA") && @printf "Iterations        = %13d.\n" iter
    (m.verbose > 0) && @printf "Total time        = %13.5f sec.\n" (time()-start)
    (m.verbose > 0) && @printf "MIP total time    = %13.5f sec.\n" cputime_mip
    (m.verbose > 0) && @printf "NLP total time    = %13.5f sec.\n\n" cputime_nlp

    (m.profile) && @printf "Profiler:\n"
    (m.profile) && @printf "Subproblem load time = %13.5f sec.\n" m.nlp_load_timer

end

MathProgBase.setwarmstart!(m::PajaritoModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PajaritoModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::PajaritoModel) = m.status
MathProgBase.getobjval(m::PajaritoModel) = m.objval
MathProgBase.getsolution(m::PajaritoModel) = m.solution
