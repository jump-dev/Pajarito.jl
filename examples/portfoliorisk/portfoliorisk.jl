# Management of a set of portfolios with overlapping stocks and constraints on different risk measures
# Choose how much to invest in each of a limited number of stocks, to maximize payoff

# Requires a special branch of JuMP to allow modeling exponential cone
using JuMP


#=========================================================
Set up JuMP model
=========================================================#

function portfoliorisk(solver, P, S, SP, Smax, returns, sigmahalf, gamma, riskball, DDT)
    m = Model(solver=solver)

    # Total investment sums to 1 (use <= 1 to simulate presence of riskless asset, ensuring feasibility)
    @variable(m, x[p in P, s in SP[p]] >= 0)
    @constraint(m, sum(x) <= 1)

    # Maximize total returns
    @objective(m, Max, sum(returns[p][s]*x[p,s] for p in P, s in SP[p]))

    # Total number of stocks with nonzero investment cannot exceed Smax (||x||_0 <= Smax)
    @variable(m, y[s in S], Bin)
    @constraint(m, sum(y) <= Smax)
    @constraint(m, [p in P, s in SP[p]], x[p,s] <= y[s])

    for p in P
        dim = length(SP[p])
        xp = [x[p,s] for s in SP[p]]
        sxp = sigmahalf[p]*xp

        if riskball[p] == :norm2
            @constraint(m, norm(sxp) <= gamma[p])
        elseif riskball[p] == :robustnorm2
            @variable(m, lambda >= 0)
            @SDconstraint(m, [gamma[p] sxp'  xp'; sxp (gamma[p] - lambda*DDT[p]) zeros(dim, dim); xp zeros(dim, dim) lambda*eye(dim))] >= 0)
        elseif riskball[p] == :entropy
            @constraint(m, sum(entropy(m, sxps) for sxps in sxp) <= gamma[p]^2)
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

    g 6n_stocks = int(readline(file))

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


por_files = readdir(joinpath(pwd(), "data")

riskball = [:norm2, :robustnorm2, :entropy]
gamma = [0.2, 0.2]
N = length(riskball)

P = 1:N
SP = Vector{Vector{String}}(N)
returns = Vector{Float64}(N)
sigmahalf = Vector{Array{Float64}}(N)
DDT = Vector{Array{Float64}}(N)
total_stocks = 0
stocks = Set{String}()

for p in P
    (n_stocks, SP[p], returns[p], sigmahalf[p]) = load_portfolio(por_files[p])

    total_stocks += n_stocks
    append!(S, SP[p])

    if riskball[p] == :robustnorm2
        # Generate perturbation matrix DDT and completely symmetrize (to avoid JuMP symmetry constraints)
        D = rand(n_stocks, round(Int, n_stocks/2))
        DDT[p] = zeros(n_stocks, n_stocks)
        for i in 1:n_stocks, j in i:n_stocks
            DDT[p][i,j] = DDT[p][j,i] = D[i]*D[j]
        end
    end
end

Smax = round(Int, total_stocks/3)
S = collect(stocks)


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

m = portfoliorisk(solver, P, S, SP, Smax, returns, sigmahalf, gamma, riskball)
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
