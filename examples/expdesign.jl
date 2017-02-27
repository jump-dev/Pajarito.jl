# These problems are numerically challenging (small epsilons etc) and cause poor stability in the MIP and conic solvers and Pajarito itself (eg negative opt gap)
# Experimental design examples from CVX, Boyd & Vandenberghe 2004 section 7.5
# http://web.cvxr.com/cvx/examples/cvxbook/Ch07_statistical_estim/html/expdesign.html

using Convex, JuMP, Pajarito
log_level = 3
mip_solver_drives = false


# using SCS
# cont_solver = SCSSolver(eps=1e-6, max_iters=1000000, verbose=0)

using Mosek
cont_solver = MosekSolver(LOG=0)


# using Cbc
# mip_solver = CbcSolver()

using CPLEX
mip_solver = CplexSolver(
    # CPX_PARAM_SCRIND=(mip_solver_drives ? 1 : 0),
    CPX_PARAM_SCRIND=1,
    CPX_PARAM_EPINT=1e-8,
    CPX_PARAM_EPRHS=1e-7,
    CPX_PARAM_EPGAP=(mip_solver_drives ? 1e-5 : 1e-8)
)


solver = PajaritoSolver(
    rel_gap=1e-5,
	mip_solver_drives=mip_solver_drives,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
	log_level=log_level,
	init_sdp_soc=true,
    sdp_soc=true,
    sdp_eig=true,
    # prim_cuts_only=true,
    # prim_cuts_always=true,
    # prim_cuts_assist=true
)



# Specify data
V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
(q, p) = size(V)
n = 7
nmax = 3



# A-optimal design
#   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
#   subject to  lambda >= 0, 1'*lambda = 1


# Convex.jl
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

# (c,A,b,cones,_) = conic_problem(aOpt)
# @show c
# @show A
# @show b
# @show cones

solve!(aOpt, solver)
println("\n  objective $(aOpt.optval)")
println("  solution\n$(np.value)")


# JuMP.jl
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

# (c, A, b, var_cones, con_cones) = JuMP.conicdata(aOpt)
# @show c
# @show A
# @show b
# @show var_cones
# @show con_cones

solve(aOpt)
println("\n  objective $(getobjectivevalue(aOpt))")
println("  solution\n$(getvalue(np))\n")



# E-optimal design
#   maximize    w
#   subject to  sum_i lambda_i*vi*vi' >= w*I
#               lambda >= 0,  1'*lambda = 1;


# Convex.jl
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

# (c,A,b,cones,_) = conic_problem(eOpt)
# @show c
# @show A
# @show b
# @show cones

solve!(eOpt, solver)
println("\n  objective $(eOpt.optval)")
println("  solution\n$(np.value)")


# JuMP.jl
println("\n\n****E optimal: JuMP.jl****\n")
eOpt = Model(solver=solver)
np = @variable(eOpt, [j=1:p], Int, lowerbound=0, upperbound=nmax)
@constraint(eOpt, sum(np) <= n)
t = @variable(eOpt)
@objective(eOpt, Max, t)
@SDconstraint(eOpt, V * diagm(np./n) * V' - t * eye(q) >= 0)

# (c, A, b, var_cones, con_cones) = JuMP.conicdata(eOpt)
# @show c
# @show A
# @show b
# @show var_cones
# @show con_cones

solve(eOpt)
println("\n  objective $(getobjectivevalue(eOpt))")
println("  solution\n$(getvalue(np))\n")



# D-optimal design
#   maximize    log det V*diag(lambda)*V'
#   subject to  sum(lambda)=1,  lambda >=0


# Convex.jl
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

# (c,A,b,cones,_) = conic_problem(dOpt)
# @show c
# @show A
# @show b
# @show cones

solve!(dOpt, solver)
println("\n  objective $(dOpt.optval)")
println("  solution\n$(np.value)")



# Find optimal solutions for each objective (D, A, E - optimal) by enumeration
using Combinatorics

DAEval = fill(Inf, 3)
DAEmin = fill(Inf, 3)
DAEvec = [Set{Vector{Int}}(), Set{Vector{Int}}(), Set{Vector{Int}}()]

for comb = [[0,1,3,3], [1,1,2,3], [1,2,2,2]]
    for perm in permutations(comb)
        E = inv(V * diagm(perm./n) * V') # Error covariance matrix

        DAEval = [det(E), trace(E), eigmax(E)] # Minimize determinant, trace, max eigenvalue

        for i in 1:3
            if DAEval[i] == DAEmin[i]
                push!(DAEvec[i], copy(perm))
                println("same min")
                @show perm
            elseif DAEval[i] < DAEmin[i]
                DAEmin[i] = DAEval[i]
                empty!(DAEvec[i])
                push!(DAEvec[i], copy(perm))
                println("new min")
                @show perm
            end
        end
    end
end

println("\nD-opt: det(E)\n$(DAEmin[1])\n$(DAEvec[1])")
println("\nA-opt: trace(E)\n$(DAEmin[2])\n$(DAEvec[2])")
println("\nE-opt: eigmax(E)\n$(DAEmin[3])\n$(DAEvec[3])")
