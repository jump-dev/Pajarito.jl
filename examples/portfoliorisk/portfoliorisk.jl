# Management of a set of portfolios with overlapping stocks and constraints on different risk measures
# Choose how much to invest in each of a limited number of stocks, to maximize payoff

# Requires a special branch of JuMP to allow modeling exponential cone
using JuMP


#=========================================================
Set up JuMP model
=========================================================#

function portfoliorisk(solver, P, S, SP, r, Smax, sigmahalf, gamma, riskball)
    m = Model(solver=solver)

    # Total investment sums to 1 (use <= 1 to simulate presence of riskless asset, ensuring feasibility)
    @variable(m, x[p in P, s in SP[p]] >= 0)
    @constraint(m, sum(x) <= 1)

    # Maximize total returns
    @objective(m, Max, sum(r[p,s]*x[p,s] for p in P, s in SP[p]))

    # Total number of stocks with nonzero investment cannot exceed Smax (||x||_0 <= Smax)
    @variable(m, y[s in S], Bin)
    @constraint(m, sum(y) <= Smax)
    @constraint(m, [p in P, s in SP[p]], x[p,s] <= y[s])

    for p in P
        sxp = sigmahalf[p]*vec(x[p,:])

        if riskball[p] == :norm2
            @constraint(m, norm(sxp) <= gamma[p])
        elseif riskball[p] == :robustnorm2

            # @SDconstraint(m, )
        elseif riskball[p] == :entropy
            @constraint(m, sum(entropy(m, sxp[s]) for s in SP[p]) <= gamma[p]^2)
        else
            error("Invalid risk ball type $(riskball[p])")
        end
    end

    return m
end

function entropy(m, q::AffExpr)
    # a + b where a >= (1+q)log(1+q), b >= (1-q)log(1-q)
    @variable(m, a)
    @Conicconstraint(m, [-a, 1 + q, 1] >= 0, :ExpPrimal)
    @variable(m, b)
    @Conicconstraint(m, [-b, 1 - q, 1] >= 0, :ExpPrimal)

    return (a + b)
end


#=========================================================
Specify/read data
=========================================================#

function load_portfolio(por_file::String)
    file = open(por_file, "r")

    n_stocks = int(readline(file))

    returns = float(split(readline(file))[1:n_stocks])

    sigmahalf = zeros(n_stocks, n_stocks)
    for s in 1:n_stocks
        sigmahalf[s,:] = float(split(readline(file))[1:n_stocks])
    end

    @assert readline(file) == ""

    nameline = readline(file)
    names = [parse(String, x) for x in split(nameline[2:end-1], ',')]

    return (n_stocks, names, returns, sigmahalf)
end


P, S, SP, r, Smax, sigmahalf, gamma, riskball



#=========================================================
Specify MICP solver
=========================================================#

using Pajarito

mip_solver_drives = true
log_level = 3
rel_gap = 1e-5

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
    # prim_cuts_always=true,
    # prim_cuts_assist=true
)


#=========================================================
Solve and print solution
=========================================================#

m = portfoliorisk(solver, P, S, SP, r, Smax, sigmahalf, gamma, riskball)
solve!(m)

@printf "\nReturns (obj) = %7.3f\n" getobjectivevalue(m)
for p in P
    @printf "\nPortfolio %s:\n" p
    for s in SP[p]
        if getvalue(y[s]) > 0.1
            @printf " %4d %7.3f\n" s getvalue(x[p,s])
        end
    end
end
