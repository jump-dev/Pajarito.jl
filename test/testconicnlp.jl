using Convex
using Pajarito
using FactCheck
using Ipopt

facts("Conic NLP Test") do

    #=m = Model(solver=IpoptSolver())
    
    c = [-2.0;0.0;0.0]
    A = [0.0 0.0 1.0;0.0 1.0 0.0]
    b = [4.0; 1.0]
    constr_cones = Any[]
    var_cones = Any[]
    push!(constr_cones, (:NonNeg,[1]))
    push!(constr_cones, (:Zero,[2]))
    push!(var_cones, (:ExpPrimal,[1 2 3]))

    x = loadconicnlpproblem(m, c, A, b, constr_cones, var_cones)

    setValue(x, [1.0;1.0;4.0])

    conic_nlp_status = solve(m)
    conic_nlp_solution = getValue(x)=#

    x = Variable(1)
    problem = maximize(x,exp(x) <= 4)
    solve!(problem, ConicNLPSolver())

    @fact evaluate(x) --> roughly(1.3862943611198906) 

end
