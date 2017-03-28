# Cardinality constrained least squares
# Cardinality constrained quadratic program
# m sample points in R^d, in matrix A in R^(m,d)
# m measurements, in vector b
# Estimate x in R^d to minimize ||Ax-b||_2
# Constrain at most k components of x to be nonzero (select k features only)
#   minimize    1/2*||Ax-b||_2^2 + 1/2*rho*||x||_2^2
#   subject to  ||x||_0 <= k
# Equivalent MISDP model described in Gally et al, 2016, section 9.2

using JuMP, Pajarito

# Set up QP JuMP model, solve, print solution
# xB is a bound on the absolute values of the estimate variables x_j
# Solver can be either MIQP/MINLP or MICP
function miqp_cardls(m, d, A, b, k, rho, xB, solver)
    mod = Model(solver=solver)
    @variable(mod, x[j in 1:d])
    @variable(mod, z[j in 1:d], Bin)
    @objective(mod, Min, 1/2*sum((A*x - b).^2) + rho/2*sum(x.^2))
    @constraint(mod, x .<= xB.*z)
    @constraint(mod, x .>= -xB.*z)
    @constraint(mod, sum(z) <= k)

    solve(mod)
    println("\n  objective $(getobjectivevalue(mod))")
    println("  solution\n$(getvalue(x))\n")
end

# Set up MISDP JuMP model, solve, print solution
# No bound xB is needed in this model
function misdp_cardls(m, d, A, b, k, rho, solver)
    mod = Model(solver=solver)
    @variable(mod, tau)
    @variable(mod, z[j in 1:d], Bin)
    @objective(mod, Min, tau)
    @constraint(mod, sum(z) <= k)
    @SDconstraint(mod, [(eye(m) + 1/rho*A*diagm(z)*A') b ; b' tau] >= 0)

    solve(mod)
    println("\n  objective $(getobjectivevalue(mod))")
    println("  solution\n$(getvalue(z))\n")
end


#=========================================================
Choose solvers and options
=========================================================#

mip_solver_drives = false
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


# Specify mixed-integer conic solver (Pajarito conic algorithm)

# using SCS
# conic_solver = SCSSolver(eps=1e-6, max_iters=1000000, verbose=0)

using Mosek
conic_solver = MosekSolver(LOG=0)

micp_solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
	mip_solver=mip_solver,
	cont_solver=conic_solver,
)


# Specify mixed-integer NLP solver (Pajarito nonlinear algorithm)

using Ipopt
nlp_solver = IpoptSolver(print_level=0)

minlp_solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
	mip_solver=mip_solver,
	cont_solver=nlp_solver,
)


#=========================================================
Specify/generate data
=========================================================#

d = 10  # Dimension of feature space
m = 30  # Number of samples

srand(100)       # Change or comment random seed to get different data
A = randn(m, d)  # Sample point matrix (rows are samples)
b = randn(m)     # Sample measurement vector
@show A
@show b

k = floor(Int, d/2)  # Number of features to select (||x||_0 <= k)

rho = 1.  # Ridge regularization multiplier

xB = 4  # Bound on absolute values of estimate variables (|x_j| <= xB)


#=========================================================
Solve JuMP models
=========================================================#

println("\n\n****MIQP model with MINLP solver****\n")
miqp_cardls(m, d, A, b, k, rho, xB, minlp_solver)

println("\n\n****MIQP model with conic solver****\n")
miqp_cardls(m, d, A, b, k, rho, xB, micp_solver)

println("\n\n****MISDP model with conic solver****\n")
misdp_cardls(m, d, A, b, k, rho, micp_solver)
