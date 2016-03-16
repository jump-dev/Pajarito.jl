#  Copyright 2016, Los Alamos National Laboratory, LANS LLC.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using JuMP

# ASSUMES LINEAR OBJECTIVE
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
    opt_tolerance               # Relatice optimality tolerance
    acceptable_opt_tolerance    # Acceptable optimality tolerance if separation fails
    time_limit                  # Time limit
    cut_switch                  # Cut level for OA
    socp_disaggregator::Bool    # SOCP disaggregator for SOC constraints
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

    # CONSTRUCTOR:
    function PajaritoModel(verbose,algorithm,mip_solver,cont_solver,opt_tolerance,acceptable_opt_tolerance,time_limit,cut_switch,socp_disaggregator,instance)
        m = new()
        m.verbose = verbose
        m.algorithm = algorithm
        m.mip_solver = mip_solver
        m.cont_solver = cont_solver
        m.opt_tolerance = opt_tolerance
        m.acceptable_opt_tolerance = acceptable_opt_tolerance
        m.time_limit = time_limit
        m.cut_switch = cut_switch
        m.socp_disaggregator = socp_disaggregator
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

MathProgBase.NonlinearModel(s::PajaritoSolver) = PajaritoModel(s.verbose, s.algorithm, s.mip_solver, s.cont_solver, s.opt_tolerance, s.acceptable_opt_tolerance, s.time_limit, s.cut_switch, s.socp_disaggregator, s.instance)

function MathProgBase.loadproblem!(
    m::PajaritoModel, numVar, numConstr, l, u, lb, ub, sense, d)

    if m.mip_solver == nothing
        error("MIP solver is not specified.")
    end

    if m.cont_solver == nothing
        error("Conic solver is not specified.")
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

function addCuttingPlanes!(m::PajaritoModel, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb)
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
                #MathProgBase.addconstr!(mip_model, varidx_new[i], coef_new[i], -Inf, new_rhs)
            else
                if cb != []
                    @addLazyConstraint(cb, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
                else
                    @addConstraint(mip_model, dot(coef_new[i], m.mip_x[varidx_new[i]]) >= new_rhs)
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
        #MathProgBase.addconstr!(mip_model, varidx, grad_f, -Inf, new_rhs)
    end


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


function MathProgBase.optimize!(m::PajaritoModel)

    start = time()

    cputime_nlp = 0.0
    cputime_mip = 0.0

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
    MathProgBase.loadproblem!(ini_nlp_model,
    m.numVar, m.numConstr, m.l, m.u, m.lb, m.ub, m.objsense, m.d)

    # pass in starting point
    #MathProgBase.setwarmstart!(nlp_model, m.solution)
    MathProgBase.setwarmstart!(ini_nlp_model, m.solution[1:m.numVar])

    start_nlp = time()
    MathProgBase.optimize!(ini_nlp_model)
    cputime_nlp += time() - start_nlp
    ini_nlp_status = MathProgBase.status(ini_nlp_model)
    if ini_nlp_status == :Infeasible || ini_nlp_status == :InfeasibleOrUnbounded
        m.status = :Infeasible
        return
    elseif ini_nlp_status == :Unbounded
        m.status = :Unbounded
        return
    end 

    ini_nlp_objval = MathProgBase.getobjval(ini_nlp_model)

    (m.verbose > 0) && println("\nPajarito started...\n")
    (m.verbose > 0) && println("MINLP algorithm $(m.algorithm) is chosen.")
    (m.verbose > 0) && println("MINLP has $(m.numVar) variables, $(m.numConstr - m.numNLConstr) linear constraints, $(m.numNLConstr) nonlinear constraints.")
    (m.verbose > 0) && println("Initial relaxation objective = $ini_nlp_objval.\n")

    separator = MathProgBase.getsolution(ini_nlp_model)
    addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, [])


    m.objval = Inf
    cut_added = false

    function nonlinearcallback(cb)
        if cb != []
            mip_objval = -Inf #MathProgBase.cbgetobj(cb)
            mip_solution = MathProgBase.cbgetmipsolution(cb)[1:m.numVar]
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
        MathProgBase.loadproblem!(nlp_model,
        m.numVar, m.numConstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)

        # pass in starting point
        #MathProgBase.setwarmstart!(nlp_model, m.solution)
        MathProgBase.setwarmstart!(nlp_model, mip_solution[1:m.numVar])

        l_inf = [new_l;zeros(m.numNLConstr)]
        u_inf = [new_u;Inf*ones(m.numNLConstr)]

        d_inf = InfeasibleNLPEvaluator(m.d, m.numConstr, m.numNLConstr, m.numVar, m.constrtype, m.constrlinear)
        inf_model = MathProgBase.NonlinearModel(m.cont_solver)
        MathProgBase.loadproblem!(inf_model,
        m.numVar+m.numNLConstr, m.numConstr, l_inf, u_inf, m.lb, m.ub, :Min, d_inf) 

        #MathProgBase.setvarUB!(nlp_model, new_u)
        #MathProgBase.setvarLB!(nlp_model, new_l)
        # optiimize the NLP problem
        start_nlp = time()
        MathProgBase.optimize!(nlp_model)
        cputime_nlp += time() - start_nlp
        nlp_status = MathProgBase.status(nlp_model)
        nlp_objval = -Inf
        inf_cut_generator = false
        #separator::Vector{Float64}
        if nlp_status == :Optimal
            (m.verbose > 2) && println("NLP Solved")
            nlp_objval = MathProgBase.getobjval(nlp_model)
            separator = MathProgBase.getsolution(nlp_model)
            (m.verbose > 2) && println("NLP Solution: $separator")

            # KEEP TRACK OF BEST KNOWN INTEGER FEASIBLE SOLUTION
            #=if m.objsense == :Max && !m.objlinear
                if nlp_objval > m.objval
                    m.objval = -nlp_objval
                    m.solution = separator[1:m.numVar]
                end
            else=#
                if nlp_objval < m.objval
                    m.objval = (m.objsense == :Max && !m.objlinear ? -nlp_objval : nlp_objval)
                    m.solution = separator[1:m.numVar]
                end 
            #end

        else
            inf_cut_generator = true
            # Create the warm start solution for inf model
            inf_initial_solution = zeros(m.numVar + m.numNLConstr);
            inf_initial_solution[1:m.numVar] = mip_solution[1:m.numVar] 
            g = zeros(m.numConstr)
            MathProgBase.eval_g(m.d, g, inf_initial_solution[1:m.numVar])
            k = 1
            for i in 1:m.numConstr
                if m.constrlinear[i]

                else
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
            if MathProgBase.status(inf_model) == :Infeasible
                (m.verbose > 2) && println("INF NLP Infeasible")
                m.status = :Infeasible
                return
            end
            (m.verbose > 2) && println("INF NLP Solved")
            separator = MathProgBase.getsolution(inf_model)
            (m.verbose > 2) && println("INF NLP Solution: $separator")
        end
        # add supporting hyperplanes
        if m.objsense == :Min
            optimality_gap = nlp_objval - mip_objval
        else
            optimality_gap = -nlp_objval - mip_objval
        end
        (m.verbose > 0) && (m.algorithm == "OA") && @printf "%9d   %13.2f   %15.2f   %14.2f   %13.2f\n" iter mip_objval nlp_objval optimality_gap m.objval
        if inf_cut_generator || optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance || cb != []
            addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f, cb)
            (m.cut_switch > 0) && addCuttingPlanes!(m, mip_model, mip_solution, jac_I, jac_J, jac_V, grad_f, cb)
            cut_added = true
        else
            (m.verbose > 1) && println("MINLP Solved")
            m.status = :Optimal
            m.objval = MathProgBase.getobjval(nlp_model)
            if m.objsense == :Max && !m.objlinear
                m.objval = -m.objval
            end
            m.solution = MathProgBase.getsolution(nlp_model)
            m.iterations = iter
            (m.verbose > 1) && println("Number of OA iterations: $iter")
            return
        end
    end

    iter = 0
    if m.algorithm == "BC"
        addLazyCallback(mip_model, nonlinearcallback)
        m.status = solve(mip_model)
    elseif m.algorithm == "OA"
        (m.verbose > 0) && println("Iteration   MIP Objective   Conic Objective   Optimality Gap   Best Solution")
        while (time() - start) < m.time_limit
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
                return
            else 
                (m.verbose > 1) && println("MIP Status: $mip_status")
            end
            mip_solution = getValue(m.mip_x)

            nonlinearcallback([])

            if cut_added == false
                break
            end

            #MathProgBase.setvarUB!(nlp_model, m.u)
            #MathProgBase.setvarLB!(nlp_model, m.l)
            iter += 1
        end
    else
        error("Unspecified algorithm.")
    end

    (m.verbose > 0) && println("\nPajarito finished...\n")
    (m.verbose > 0) && println("Status = $(m.status).")
    (m.verbose > 0) && println("Total time = $(time() - start) sec. Iterations = $iter.") 
    (m.verbose > 0) && println("MIP total time = $(cputime_mip).")
    (m.verbose > 0) && println("NLP total time = $(cputime_nlp).")
    (m.verbose > 0) && (m.status == :Optimal) && println("Optimum objective = $(m.objval).\n") 

end

MathProgBase.setwarmstart!(m::PajaritoModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PajaritoModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::PajaritoModel) = m.status
MathProgBase.getobjval(m::PajaritoModel) = m.objval
MathProgBase.getsolution(m::PajaritoModel) = m.solution
