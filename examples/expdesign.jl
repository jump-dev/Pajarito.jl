# Experimental design examples (D-optimal, A-optimal, and E-optimal) from Boyd and Vandenberghe, "Convex Optimization", section 7.5
# CVX code adapted from expdesign.m by Lieven Vandenberghe and Almir Mutapcic
# http://cvxr.com/cvx/examples/cvxbook/Ch07_statistical_estim/html/expdesign.html
#
# A-optimal design
#   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
#   subject to  lambda >= 0, 1'*lambda = 1
#
# E-optimal design
#   maximize    w
#   subject to  sum_i lambda_i*vi*vi' >= w*I
#               lambda >= 0,  1'*lambda = 1;
#
# D-optimal design
#   maximize    log det V*diag(lambda)*V'
#   subject to  sum(lambda)=1,  lambda >=0

using Convex, JuMP, Pajarito

# Generate random V (experimental matrix) data
# q is dimension of estimate space
# p is number of experimental vectors
function gen_V(q, p, V)
    for i in 1:q, j in 1:p
        v = randn()
        if abs(v) < 1e-2
            v = 0.
        end
        V[i, j] = v
    end
    return V
end

# A Optimal: Convex.jl
function aOpt_Convexjl(q, p, V, n, nmax)
    println("\n\n****A optimal: Convex.jl****\n")

    np = Convex.Variable(p, :Int)
    Q = Convex.Variable(q, q)
    u = Convex.Variable(q)
    aOpt = minimize(
        sum(u),
        Q == V * diagm(np./n) * V',
        sum(np) <= n,
    	np >= 0,
    	np <= nmax
    )
    E = eye(q)
    for i in 1:q
    	aOpt.constraints += isposdef([Q E[:,i]; E[i,:]' u[i]])
    end

    solve!(aOpt, solver)
    println("\n  objective $(aOpt.optval)")
    println("  solution\n$(np.value)")
end

# A Optimal: JuMP.jl
function aOpt_JuMPjl(q, p, V, n, nmax)
    println("\n\n****A optimal: JuMP.jl****\n")

    aOpt = Model(solver=solver)
    np = @variable(aOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    @constraint(aOpt, sum(np) <= n)
    u = @variable(aOpt, [i=1:q], lowerbound=0)
    @objective(aOpt, Min, sum(u))
    E = eye(q)
    for i=1:q
        @SDconstraint(aOpt, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
    end

    solve(aOpt)
    println("\n  objective $(getobjectivevalue(aOpt))")
    println("  solution\n$(getvalue(np))\n")
end

# E Optimal: Convex.jl
function eOpt_Convexjl(q, p, V, n, nmax)
    println("\n\n****E optimal: Convex.jl****\n")

    np = Convex.Variable(p, :Int)
    Q = Convex.Variable(q, q)
    t = Convex.Variable()
    eOpt = maximize(
        t,
        Q == V * diagm(np./n) * V',
        sum(np) <= n,
    	np >= 0,
    	np <= nmax,
        isposdef(Q - t * eye(q))
    )

    solve!(eOpt, solver)
    println("\n  objective $(eOpt.optval)")
    println("  solution\n$(np.value)")
end

# E Optimal: JuMP.jl
function eOpt_JuMPjl(q, p, V, n, nmax)
    println("\n\n****E optimal: JuMP.jl****\n")

    eOpt = Model(solver=solver)
    np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
    @constraint(eOpt, sum(np) <= n)
    t = @variable(eOpt)
    @objective(eOpt, Max, t)
    @SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

    solve(eOpt)
    println("\n  objective $(getobjectivevalue(eOpt))")
    println("  solution\n$(getvalue(np))\n")
end

# D Optimal: Convex.jl
function dOpt_Convexjl(q, p, V, n, nmax)
    println("\n\n****D optimal: Convex.jl****\n")

    np = Convex.Variable(p, :Int)
    Q = Convex.Variable(q, q)
    dOpt = maximize(
        logdet(Q),
        Q == V * diagm(np./n) * V',
        sum(np) <= n,
        np >= 0,
        np <= nmax
    )

    solve!(dOpt, solver)
    println("\n  objective $(dOpt.optval)")
    println("  solution\n$(np.value)")
end

# D Optimal: JuMP.jl model
# Not available until JuMP handles exponential cones


#=========================================================
Choose solvers and options
=========================================================#

mip_solver_drives = true
log_level = 3
rel_gap = 1e-5

# using Cbc
# mip_solver = CbcSolver()

using CPLEX
mip_solver = CplexSolver(
    CPX_PARAM_SCRIND=(mip_solver_drives ? 1 : 0),
    # CPX_PARAM_SCRIND=1,
    CPX_PARAM_EPINT=1e-8,
    CPX_PARAM_EPRHS=1e-7,
    CPX_PARAM_EPGAP=(mip_solver_drives ? 1e-5 : 1e-9)
)

using SCS
cont_solver = SCSSolver(eps=1e-6, max_iters=1000000, verbose=1)

# using Mosek
# cont_solver = MosekSolver(LOG=0)

solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
    solve_subp=true,
    solve_relax=true,
	init_sdp_soc=false,
    sdp_soc=false,
    sdp_eig=true,
    # prim_cuts_only=true,
    prim_cuts_always=true,
    # prim_cuts_assist=true
)


#=========================================================
Specify/generate data
=========================================================#

# Uncomment only the line with the desired data dimensions
(q, p, n, nmax) = (
    # 100, 250, 500, 5   # Huge
    # 25, 75, 125, 5     # Large
    # 10, 30, 50, 5      # Medium
    # 5, 15, 25, 5       # Small
    4, 8, 12, 3        # Tiny
)
@assert (p > q) && (n > q)

# Generate matrix of experimental vectors
# Change or comment random seed to get different random V matrix
srand(100)

V = Array{Float64}(q, p)
tries = 0
while true
    # Generate random V
    V = gen_V(q, p, V)
    tries += 1

    # Ensure rank is q
    if rank(V) == q
        break
    end

    if tries > 100
        error("Could not generate random V matrix with rank equal to q in 100 tries\n")
    end
end
# @show tries
# @show V


#=========================================================
Solve Convex.jl models
=========================================================#

aOpt_Convexjl(q, p, V, n, nmax)
eOpt_Convexjl(q, p, V, n, nmax)
dOpt_Convexjl(q, p, V, n, nmax)


#=========================================================
Solve JuMP.jl models
=========================================================#

aOpt_JuMPjl(q, p, V, n, nmax)
eOpt_JuMPjl(q, p, V, n, nmax)
