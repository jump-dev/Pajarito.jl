######################################################
# This package contains the mixed-integer non-linear
# programming (MINLP) problem solver Pajarito.jl:
#
#       P olyhedral
#       A pproximation
# (in)  J ulia :
#       A utomatic
#       R eformulations
# (for) I n T eger
#       O ptimization
# 
# It applies outer approximation to a series of
# mixed-integer linear programming problems
# that approximates the original MINLP in a polyhedral
# form.
######################################################

# ASSUMES LINEAR OBJECTIVE
type PajaritoModel <: MathProgBase.AbstractNonlinearModel
    solution::Vector{Float64}
    status
    objval::Float64
    iterations::Int
    numVar::Int
    numConstr::Int
    numNLConstr::Int
    verbose::Int
    mip_solver
    nlp_solver
    opt_tolerance
    time_limit
    cut_switch
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
    d
    function PajaritoModel(verbose,mip_solver,nlp_solver,opt_tolerance,time_limit,cut_switch)
        m = new()
        m.verbose = verbose
        m.mip_solver = mip_solver
        m.nlp_solver = nlp_solver
        m.opt_tolerance = opt_tolerance
        m.time_limit = time_limit
        m.cut_switch = cut_switch
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


export PajaritoSolver
immutable PajaritoSolver <: MathProgBase.AbstractMathProgSolver
    verbose
    mip_solver
    nlp_solver
    opt_tolerance
    time_limit
    cut_switch
end
PajaritoSolver(;verbose=0,mip_solver=CplexSolver(CPX_PARAM_SCRIND=0,CPX_PARAM_REDUCE=0,CPX_PARAM_EPINT=1e-8),nlp_solver=IpoptSolver(print_level=0),opt_tolerance=1e-6,time_limit=60*60,cut_switch=1) = PajaritoSolver(verbose,mip_solver,nlp_solver,opt_tolerance,time_limit,cut_switch)

# BEGIN MATHPROGBASE INTERFACE

MathProgBase.NonlinearModel(s::PajaritoSolver) = PajaritoModel(s.verbose, s.mip_solver, s.nlp_solver, s.opt_tolerance, s.time_limit, s.cut_switch)

function MathProgBase.loadproblem!(
    m::PajaritoModel, numVar, numConstr, l, u, lb, ub, sense, d)

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

    m.A = sparse(A_I, A_J, A_V)

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

function addCuttingPlanes!(m::PajaritoModel, mip_model, separator, jac_I, jac_J, jac_V, grad_f)
    # EVALUATE g and jac_g AT MIP SOLUTION THAT IS INFEASIBLE
    g = zeros(m.numConstr)
    MathProgBase.eval_g(m.d, g, separator[1:m.numVar])
    MathProgBase.eval_jac_g(m.d, jac_V, separator[1:m.numVar])

    # create rows corresponding to constraints in sparse format

    varidx_new = [zeros(0) for i in 1:m.numConstr]
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
                MathProgBase.addconstr!(mip_model, varidx_new[i], coef_new[i], -Inf, new_rhs)
            else
                MathProgBase.addconstr!(mip_model, varidx_new[i], coef_new[i], new_rhs, Inf)
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
        varidx = zeros(m.numVar+1)
        for j = 1:m.numVar
            varidx[j] = j
            new_rhs += grad_f[j] * separator[j]
        end
        varidx[m.numVar+1] = m.numVar+1
        grad_f[m.numVar+1] = -(1.0)
        (m.verbose > 1) && println("varidx $(varidx)") 
        (m.verbose > 1) && println("coef $(grad_f)") 
        (m.verbose > 1) && println("rhs $new_rhs") 
        MathProgBase.addconstr!(mip_model, varidx, grad_f, -Inf, new_rhs)
    end


end

function MathProgBase.optimize!(m::PajaritoModel)

    start = time()

    populatelinearmatrix(m)
    # solve it
    # MathProgBase.optimize!(nlp_model)

    # pull out solution values
    #= m.status = MathProgBase.status(nlp_model)
    m.objval = MathProgBase.getobjval(nlp_model)
    m.solution = MathProgBase.getsolution(nlp_model) =#

    # MODIFICATION new objective t >= f(x)
    # set up cplex to solve the mixed integer linear problem
    mip_model = MathProgBase.LinearQuadraticModel(m.mip_solver)
    MathProgBase.loadproblem!(mip_model,
        [m.A spzeros(size(m.A,1),1)],
        [m.l; -1e4],
        [m.u; Inf],
        [(m.objsense == :Max)? -m.c : m.c; m.objlinear? 0.0 : 1.0], m.A_lb, m.A_ub, :Min)
    MathProgBase.setvartype!(mip_model, [m.vartype; :Cont])

    for i in 1:m.numConstr
        if !(m.constrlinear[i]) && m.constrtype[i] == :(==)
            error("Nonlinear equality or two-sided constraints not accepted.")
        end
    end


    jac_I, jac_J = MathProgBase.jac_structure(m.d)
    jac_V = zeros(length(jac_I))
    grad_f = zeros(m.numVar+1)

    ini_nlp_model = MathProgBase.NonlinearModel(m.nlp_solver)
    MathProgBase.loadproblem!(ini_nlp_model,
    m.numVar, m.numConstr, m.l, m.u, m.lb, m.ub, m.objsense, m.d)

    # pass in starting point
    #MathProgBase.setwarmstart!(nlp_model, m.solution)
    MathProgBase.setwarmstart!(ini_nlp_model, m.solution[1:m.numVar])
    MathProgBase.optimize!(ini_nlp_model)
    ini_nlp_status = MathProgBase.status(ini_nlp_model)
    if ini_nlp_status == :Infeasible || ini_nlp_status == :InfeasibleOrUnbounded
        m.status = :Infeasible
        return
    elseif ini_nlp_status == :Unbounded
        m.status = :Unbounded
        return
    end 
    separator = MathProgBase.getsolution(ini_nlp_model)
    addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f)

    iter = 0
    while (time() - start) < m.time_limit
        # solve MIP model
        MathProgBase.optimize!(mip_model)
        mip_status = MathProgBase.status(mip_model)
        #mip_objval = Inf
        #mip_solution = zeros(m.numVar+1)
        if mip_status == :Infeasible || mip_status == :InfeasibleOrUnbounded
            (m.verbose > 0) && println("MIP Infeasible")
            m.status = :Infeasible
            return
        else 
            (m.verbose > 0) && println("MIP Status: $mip_status")
        end
        mip_objval = MathProgBase.getobjval(mip_model)
        mip_solution = MathProgBase.getsolution(mip_model)
        (m.verbose > 1) && println("MIP Solution: $mip_solution")
        # solve NLP model for the MIP solution
        new_u = m.u
        new_l = m.l
        for i in 1:m.numVar
            @show i, m.vartype[i]
            if m.vartype[i] == :Int || m.vartype[i] == :Bin
                new_u[i] = mip_solution[i]
                new_l[i] = mip_solution[i]
            end   
        end

        # set up ipopt to solve continuous relaxation
        nlp_model = MathProgBase.NonlinearModel(m.nlp_solver)
        MathProgBase.loadproblem!(nlp_model,
        m.numVar, m.numConstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)

        # pass in starting point
        #MathProgBase.setwarmstart!(nlp_model, m.solution)
        MathProgBase.setwarmstart!(nlp_model, mip_solution[1:m.numVar])

        l_inf = [new_l;zeros(m.numNLConstr)]
        u_inf = [new_u;Inf*ones(m.numNLConstr)]

        d_inf = InfeasibleNLPEvaluator(m.d, m.numConstr, m.numNLConstr, m.numVar, m.constrtype, m.constrlinear)
        inf_model = MathProgBase.NonlinearModel(m.nlp_solver)
        MathProgBase.loadproblem!(inf_model,
        m.numVar+m.numNLConstr, m.numConstr, l_inf, u_inf, m.lb, m.ub, :Min, d_inf) 

        #MathProgBase.setvarUB!(nlp_model, new_u)
        #MathProgBase.setvarLB!(nlp_model, new_l)
        # optiimize the NLP problem
        MathProgBase.optimize!(nlp_model)
        nlp_status = MathProgBase.status(nlp_model)
        nlp_objval = -Inf
        inf_cut_generator = false
        #separator::Vector{Float64}
        if nlp_status == :Optimal
            (m.verbose > 0) && println("NLP Solved")
            nlp_objval = MathProgBase.getobjval(nlp_model)
            separator = MathProgBase.getsolution(nlp_model)
            (m.verbose > 1) && println("NLP Solution: $separator")
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
            (m.verbose > 0) && println("NLP Infeasible")
            MathProgBase.optimize!(inf_model)
            if MathProgBase.status(inf_model) == :Infeasible
                (m.verbose > 0) && println("INF NLP Infeasible")
                m.status = :Infeasible
                return
            end
            (m.verbose > 0) && println("INF NLP Solved")
            separator = MathProgBase.getsolution(inf_model)
            (m.verbose > 1) && println("INF NLP Solution: $separator")
        end
        # add supporting hyperplanes
        if m.objsense == :Min
            optimality_gap = nlp_objval - mip_objval
        else
            optimality_gap = -nlp_objval - mip_objval
        end
        (m.verbose > 0) && println("Optimality Gap: $(nlp_objval) - $(mip_objval) = $(optimality_gap)")
        if inf_cut_generator || optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance
            addCuttingPlanes!(m, mip_model, separator, jac_I, jac_J, jac_V, grad_f)
            (m.cut_switch > 0) && addCuttingPlanes!(m, mip_model, mip_solution, jac_I, jac_J, jac_V, grad_f)
        else
            (m.verbose > 0) && println("MINLP Solved")
            m.status = :Optimal
            m.objval = MathProgBase.getobjval(nlp_model)
            if m.objsense == :Max && !m.objlinear
                m.objval = -m.objval
            end
            m.solution = MathProgBase.getsolution(nlp_model)
            m.iterations = iter
            (m.verbose > 0) && println("Number of OA iterations: $iter")
            return
        end
        #MathProgBase.setvarUB!(nlp_model, m.u)
        #MathProgBase.setvarLB!(nlp_model, m.l)
        iter += 1
    end
    m.status = :UserLimit
end

MathProgBase.setwarmstart!(m::PajaritoModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PajaritoModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::PajaritoModel) = m.status
MathProgBase.getobjval(m::PajaritoModel) = m.objval
MathProgBase.getsolution(m::PajaritoModel) = m.solution
