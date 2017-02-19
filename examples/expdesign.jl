# These problems are numerically challenging (small epsilons etc) and cause poor stability in the MIP and conic solvers and Pajarito itself (eg negative opt gap)
# Experimental design examples from CVX, Boyd & Vandenberghe 2004 section 7.5
# http://web.cvxr.com/cvx/examples/cvxbook/Ch07_statistical_estim/html/expdesign.html

using Convex, Pajarito
log_level = 2

using SCS
cont_solver = SCSSolver(eps=1e-5, max_iters=1000000, verbose=0)

# using Mosek
# cont_solver = MosekSolver(LOG=0)

# using Cbc
# mip_solver = CbcSolver()
# mip_solver_drives = false

using CPLEX
mip_solver = CplexSolver(CPX_PARAM_SCRIND=1, CPX_PARAM_EPINT=1e-8, CPX_PARAM_EPRHS=1e-6, CPX_PARAM_EPGAP=0.)
mip_solver_drives = true


solver = PajaritoSolver(
	mip_solver_drives=mip_solver_drives,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
	log_level=log_level,
	sdp_eig=true,
	init_sdp_lin=true,
)


enforce_integrality = true

n = 15

# Use a matrix of values generated similarly to Boyd & Vandenberghe
# Likely to cause numerical difficulties
# m = 4
# angles1 = linspace(3/4*pi, pi, m)
# angles2 = linspace(-3/8*pi, -5/8*pi, m)
# angles3 = linspace(-1/6*pi, 1/4*pi, m)
# V = [
#     3.*cos(angles1)' 1.8.*cos(angles2)' 1.*cos(angles3)';
#     3.*sin(angles1)' 1.8.*sin(angles2)' 1.*sin(angles3)';
#     3.*cos(angles2)' 1.8.*cos(angles3)' 1.*cos(angles1)';
#     3.*sin(angles2)' 1.8.*sin(angles3)' 1.*sin(angles1)'
#     ]
# V = trunc(V, 3)
# (q, p) = size(V)

# Use a random matrix of integers in (-10, 10)
# q = 5
# p = 6
# V = round.(20 .* rand(q, p) .- 10)

V = [-6.0 -3.0 8.0 3.0; -3.0 -9.0 -4.0 3.0; 3.0 1.0 5.0 5.0]
(q, p) = size(V)


np = enforce_integrality ? Variable(p, :Int) : Variable(p)


# D-optimal design
#   maximize    nthroot det V*diag(lambda)*V'
#   subject to  sum(lambda)=1,  lambda >=0
println("\n\n****D optimal****\n")
dOpt = maximize(
    logdet(V * diagm(np./n) * V'),
    sum(np) <= n,
    np >= 0
)
solve!(dOpt, (enforce_integrality ? solver : cont_solver))
println("  objective $(dOpt.optval)")
println("  solution\n$(np.value)")

# A-optimal design
#   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
#   subject to  lambda >= 0, 1'*lambda = 1
println("\n\n****A optimal****\n")
u = Variable(q)
aOpt = minimize(
    sum(u),
    sum(np) <= n,
    np >= 0
)
E = eye(q)
for i in 1:q
	aOpt.constraints += isposdef([V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]])
end
solve!(aOpt, (enforce_integrality ? solver : cont_solver))
println("  objective $(aOpt.optval)")
println("  solution\n$(np.value)")

# E-optimal design
#   maximize    w
#   subject to  sum_i lambda_i*vi*vi' >= w*I
#               lambda >= 0,  1'*lambda = 1;
println("\n\n****E optimal****\n")
t = Variable()
eOpt = maximize(
    t,
    sum(np) <= n,
    np >= 0,
    isposdef(V * diagm(np./n) * V' - t * eye(q))
)
solve!(eOpt, (enforce_integrality ? solver : cont_solver))
println("  objective $(eOpt.optval)")
println("  solution\n$(np.value)")
