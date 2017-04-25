# Management of a set of portfolios with overlapping stocks and constraints on different risk measures
# Choose how much to invest in each of a limited number of stocks, to maximize payoff

# Requires a special branch of JuMP to allow modeling exponential cone
using JuMP


#=========================================================
JuMP model functions
=========================================================#

function portfoliorisk(solver, P, S, SP, Smax, returns, Sigmahalf, gamma, riskball, Delta)
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
        sxp = Sigmahalf[p]*xp

        if riskball[p] == :norm2
            @constraint(m, norm(sxp) <= gamma[p])
        elseif riskball[p] == :robustnorm2
            lambda = @variable(m)
            @SDconstraint(m, [gamma[p] sxp'  xp'; sxp (gamma[p] - lambda*Delta[p]) zeros(dim, dim); xp zeros(dim, dim) lambda*eye(dim))] >= 0)
        elseif riskball[p] == :entropy
            @constraint(m, sum(entropy(m, sxps) for sxps in sxp) <= gamma[p]^2)
        else
            error("Invalid risk ball type $(riskball[p])")
        end
    end

    return (m, x, y)
end

function entropy(m, q::AffExpr)
    # a + b where a >= (1+q)log(1+q), b >= (1-q)log(1-q)
    a = @variable(m)
    @Conicconstraint(m, [-a, 1 + q, 1] >= 0, :ExpPrimal)
    b = @variable(m)
    @Conicconstraint(m, [-b, 1 - q, 1] >= 0, :ExpPrimal)

    return (a + b)
end


#=========================================================
Data generation functions
=========================================================#

# Generate model data from basic model options, reading portfolio data from portfoliofiles
function generatedata(balls, counts, maxstocks, gammas, Smax, portfoliofiles)
    N = sum(counts)
    @printf "\n\nGenerating data for %d portfolios\n" N
    @printf "\n%6s %6s %12s:" "Name" "Stocks" "Risk type"

    P = 1:N
    SP = Vector{Vector{String}}(N)
    returns = Vector{Float64}(N)
    Sigmahalf = Vector{Array{Float64}}(N)
    riskball = Vector{Symbol}(N)
    gamma = Vector{Float64}(N)
    Delta = Vector{Array{Float64}}(N)

    stockset = Set{String}()
    p = 0
    for b in 1:length(balls), bp in 1:counts[b] # for each ball type, each portfolio of the ball type
        p += 1

        riskball[p] = balls[b]
        gamma[p] = gammas[b]

        (returns[p], Sigmahalf[p], SP[p]) = loadportfolio(portfoliofiles[p], maxstocks[b]) # read raw data for portfolio
        numstocks = length(SP[p])
        append!(stockset, SP[n])

        if riskball[n] == :robustnorm2
            # Generate random matrix and scale and clean zeros
            Deltahalf = randn(numstocks, numstocks)
            scalefactor = 1/2*norm(Sigmahalf[n])/norm(Deltahalf)
            @assert 1e-2 < scalefactor < 1e2
            for i in 1:numstocks, j in 1:numstocks
                val = scalefactor*Deltahalf[i,j]
                if val < 1e-3
                    Deltahalf[i,j] = 0.
                else
                    Deltahalf[i,j] = val
                end
            end

            # Fill Delta matrix and completely symmetrize (to avoid JuMP adding PSD symmetry constraints)
            Delta[p] = zeros(numstocks, numstocks)
            for i in 1:numstocks, j in i:numstocks
                Delta[p][i,j] = Delta[p][j,i] = vecdot(Deltahalf[i,:], Deltahalf[:,j])
            end
        end

        @printf "\n%6d %6d %12s" p numstocks string(riskball[n])
    end

    S = collect(stockset)

    @printf "\n\nChoose %d of %d unique stocks (sum of portfolio sizes is %d)\n\n" Smax length(S) sum(length.(SP))

    return (P, SP, S, Smax, returns, Sigmahalf, riskball, gamma, Delta)
end

# Load data from a .por file, returning returns, sqrt of covariance matrix, ticker names; take at most maxstocks stocks
function loadportfolio(portfoliofile::String, maxstocks::Int)
    file = open(portfoliofile, "r")

    n = parse(Int, chomp(readline(file)))

    if n > maxstocks
        taken = maxstocks
        takestocks = permute(1:n)[1:taken]
    else
        taken = n
        takestocks = permute(1:n)
    end

    data = split(chomp(readline(file)))
    @assert length(data) == n
    rawreturns = [parse(Float64, data[s]) for s in takestocks]

    matdata = zeros(n, n)
    for i in 1:n
        data = split(chomp(readline(file)))
        @assert length(data) == n
        for j in 1:n
            matdata[i,j] = parse(Float64, data[j])
        end
    end
    rawSigmahalf = matdata[takestocks,takestocks]

    @assert chomp(readline(file)) == ""

    line = chomp(readline(file))
    @assert startswith(line, '[') && endswith(line, ']')
    data = split(line[2:end-1], ',')
    @assert length(data) == n
    rawnames = [parse(String, data[s]) for s in takestocks]

    return (rawreturns, rawSigmahalf, rawnames)
end


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
Specify model options and generate data
=========================================================#

balls = [:norm2, :robustnorm2, :entropy]
counts = [5, 2, 5]
maxstocks = [50, 8, 30]
gammas = [0.2, 0.3, 0.25]

Smax = round(Int, sum(counts)/3)

portfoliofiles = readdir(joinpath(pwd(), "data")

(P, SP, S, Smax, returns, Sigmahalf, riskball, gamma, Delta) = generatedata(balls, counts, maxstocks, gammas, Smax, portfoliofiles)


#=========================================================
Solve and print solution
=========================================================#

(m, x, y) = portfoliorisk(solver, P, SP, S, Smax, returns, Sigmahalf, riskball, gamma, Delta)

solve!(m)

@printf "\nReturns (obj) = %7.3f\n" getobjectivevalue(m)
for p in P
    @printf "\nPortfolio %d investments:\n" p
    for s in SP[p]
        if getvalue(y[s]) > 0.1
            @printf "%6d %8.4f\n" s getvalue(x[p,s])
        end
    end
end
