type PajaritoFeasibilityModel <: MathProgBase.AbstractNonlinearModel
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
    function PajaritoFeasibilityModel(verbose,mip_solver,nlp_solver,opt_tolerance,time_limit,cut_switch)
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

export PajaritoFeasibilitySolver
immutable PajaritoFeasibilitySolver <: MathProgBase.AbstractMathProgSolver
    verbose
    mip_solver
    nlp_solver
    opt_tolerance
    time_limit
    cut_switch
end
PajaritoFeasibilitySolver(;verbose=0,mip_solver=CplexSolver(CPX_PARAM_SCRIND=0,CPX_PARAM_REDUCE=0,CPX_PARAM_EPINT=1e-8),nlp_solver=IpoptSolver(print_level=0),opt_tolerance=1e-6,time_limit=60*60,cut_switch=1) = PajaritoFeasibilitySolver(verbose,mip_solver,nlp_solver,opt_tolerance,time_limit,cut_switch)

# BEGIN MATHPROGBASE INTERFACE

MathProgBase.NonlinearModel(s::PajaritoFeasibilitySolver) = PajaritoFeasibilityModel(s.verbose, s.mip_solver, s.nlp_solver, s.opt_tolerance, s.time_limit, s.cut_switch)

function MathProgBase.loadproblem!(
    m::PajaritoFeasibilityModel, numVar, numConstr, l, u, lb, ub, sense, d)

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

function evaluateConstraints(m::PajaritoFeasibilityModel, x)
    function_val = zeros(m.numConstr)        
    constr_val = zeros(m.numConstr)        
    MathProgBase.eval_g(m.d, function_val, x)
    for i in 1:m.numConstr
        if m.constrtype[i] == :(<=)
            constr_val[i] = function_val[i] - m.ub[i]
        else
            constr_val[i] = m.lb[i] - function_val[i]
        end
    end
    
    max_constr_viol = -1e-10
    for i in 1:m.numConstr
        if m.constrtype[i] == :(<=) || m.constrtype[i] == :(>=)
            viol = constr_val[i]
        else
            viol = abs(constr_val[i])
        end
        if viol > max_constr_viol
            max_constr_viol = viol
        end
    end
    @show max_constr_viol
end

function MathProgBase.optimize!(m::PajaritoFeasibilityModel)


            input_order_file = open("gams01.col")
            order_lines = readlines(input_order_file)
            var_map = Dict()
            var_names = String[]
            for i in 1:m.numVar
                var_name = order_lines[i]
                var_map[var_name[1:end-1]] = i
            end
            @show var_map 
            check_solution = zeros(m.numVar) 

            input_file = open("solution.txt","r")
            lines = readlines(input_file)
            mip_solution = zeros(m.numVar)
            @show length(lines), m.numVar
            elements = split(lines[1])
            mip_objval = float(elements[3])
            for i in 1:m.numVar
                elements = split(lines[i+1])
                mip_solution[i] = float(elements[3])
                check_solution[var_map[elements[1]]] = mip_solution[i]
                push!(var_names, elements[1])
                if mip_solution[i] != 0.0
                    @show elements[1], mip_solution[i]
                end 
            end
            close(input_file)

        evaluateConstraints(m, check_solution)

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
        nlp_model = MathProgBase.model(m.nlp_solver)
        MathProgBase.loadnonlinearproblem!(nlp_model,
        m.numVar, m.numConstr, new_l, new_u, m.lb, m.ub, m.objsense, m.d)

        # pass in starting point
        #MathProgBase.setwarmstart!(nlp_model, m.solution)
        MathProgBase.setwarmstart!(nlp_model, mip_solution[1:m.numVar])

        MathProgBase.optimize!(nlp_model)
        nlp_status = MathProgBase.status(nlp_model)
        nlp_objval = -Inf
        #separator::Vector{Float64}
        if nlp_status == :Optimal
            (m.verbose > 0) && println("NLP Solved")
            nlp_objval = MathProgBase.getobjval(nlp_model)
            separator = MathProgBase.getsolution(nlp_model)
            (m.verbose > 1) && println("NLP Solution: $separator")
        end
        if m.objsense == :Min
            optimality_gap = nlp_objval - mip_objval
        else
            optimality_gap = -nlp_objval - mip_objval
        end
        (m.verbose > 0) && println("Optimality Gap: $(nlp_objval) - $(mip_objval) = $(optimality_gap)")
        if optimality_gap > (abs(mip_objval) + 1e-5)*m.opt_tolerance
            (m.verbose > 0) && println("MINLP NOT OPTIMAL")
        else
            (m.verbose > 0) && println("MINLP Solved")
            m.status = :Optimal
            m.objval = MathProgBase.getobjval(nlp_model)
            if m.objsense == :Max && !m.objlinear
                m.objval = -m.objval
            end
            m.solution = MathProgBase.getsolution(nlp_model)
            evaluateConstraints(m, m.solution)
            output_file = open("solution_gams01.txt", "w")
            write(output_file, "objvar $(m.objval)\n")
            for i in 1:m.numVar
                var_val = m.solution[var_map[var_names[i]]]
                if var_val > 0.0
                    write(output_file, "$(var_names[i]) $var_val\n")
                end
            end
            close(output_file)
            return
        end
end

MathProgBase.setwarmstart!(m::PajaritoFeasibilityModel, x) = (m.solution = x)
MathProgBase.setvartype!(m::PajaritoFeasibilityModel, v::Vector{Symbol}) = (m.vartype = v)
#MathProgBase.setvarUB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.u = v)
#MathProgBase.setvarLB!(m::IpoptMathProgModel, v::Vector{Float64}) = (m.l = v)

MathProgBase.status(m::PajaritoFeasibilityModel) = m.status
MathProgBase.getobjval(m::PajaritoFeasibilityModel) = m.objval
MathProgBase.getsolution(m::PajaritoFeasibilityModel) = m.solution
