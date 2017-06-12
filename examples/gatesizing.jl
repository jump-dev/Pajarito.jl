# Electrical circuit gate sizing example adapted from Boyd, Kim, Vandenberghe, and Hassibi, "A Tutorial on Geometric Programming", p28
# CVX code adapted from simple_dig_ckt_sizing.m by Almir Mutapcic
# http://cvxr.com/cvx/examples/gp_tutorial/html/simple_dig_ckt_sizing.html
#
# Choose gate scale factors y_i to minimize ckt delay, subject to limits on the total area and power
#   minimize     D
#   subject to   P <= Pmax,
#                A <= Amax
#                y >= 1
# Where variables are scale factors y, which are constrained to equal discrete values
# Data is hard-coded from the circuit shown in figure 4 of "A Tutorial on Geometric Programming"

using Convex, Pajarito

# Set up Convex.jl model, solve, print solution
function gatesizing(yUB)
    fe = [1, 0.8, 1, 0.7, 0.7, 0.5, 0.5] .* [1, 2, 1, 1.5, 1.5, 1, 2]
    Cout6 = 10
    Cout7 = 10

    y = Variable(7, Positive())
    z = Variable(yUB, 7, Positive(), :Bin)

    D1 = exp(-y[1]) + exp(-y[1] + y[4])
    D2 = 2 * exp(-y[2]) + exp(-y[2] + y[4]) + exp(-y[2] + y[5])
    D3 = 2 * exp(-y[3]) + exp(-y[3] + y[5]) + exp(-y[3] + y[7])
    D4 = 2 * exp(-y[4]) + exp(-y[4] + y[6]) + exp(-y[4] + y[7])
    D5 = exp(-y[5]) + exp(-y[5] + y[7])
    D6 = Cout6 * exp(-y[6])
    D7 = Cout7 * exp(-y[7])

    P = minimize(
        maximum([(D1+D4+D6), (D1+D4+D7), (D2+D4+D6), (D2+D4+D7), (D2+D5+D7), (D3+D5+D6), (D3+D7)]),
        sum(fe .* exp(y)) <= 20,
        sum(exp(y)) <= 100)

    for i in 1:7
        P.constraints += (sum(z[:,i]) == 1)
        P.constraints += (y[i] == sum([log(j) * z[j,i] for j=1:yUB]))
    end

    solve!(P, solver)

    println("\nCircuit delay (obj) = $(P.optval)")
    println("Scale factors (exp(y)):\n$(exp.(y.value))")
    println("Value indicators (z):\n$(round.(z.value))")
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


using SCS
cont_solver = SCSSolver(eps=1e-5, max_iters=1000000, verbose=1)

# using ECOS
# cont_solver = ECOSSolver(verbose=false)


solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
    init_exp=true,
)


#=========================================================
Specify data
=========================================================#

# Integer upper bound on integer gate size factors
# Optimal solution for this instances has all exp(y) <= 3,
# so yUB larger than 2 is not constraining, but impacts number of variables
yUB = 3


#=========================================================
Solve Convex.jl model
=========================================================#

# Optimal solution is [2,3,3,3,2,3,3] with value 8.3333
gatesizing(yUB)
