# Management of a set of portfolios with overlapping stocks and constraints on different risk measures
# Choose how much to invest in each of a limited number of stocks, to maximize payoff

mutable struct Portfolio
    id::Int
    size::Int
    stocks::Vector{String}
    returns::Dict{String,Float64}
    Sigmahalf::Dict{Tuple{String,String},Float64}
    risk::Symbol
    gamma::Float64
    Delta::Dict{Tuple{String,String},Float64}
end

function Base.show(io::IO, p::Portfolio)
    print(io, p.id)
end


#=========================================================
JuMP model functions
=========================================================#

# If using entropy constraints, requires a special branch of JuMP to allow modeling exponential cone (https://github.com/chriscoey/JuMP.jl/tree/coneconstr)
using JuMP

function portfoliorisk(solver, Portfolios, Stocks, Smax)
    m = Model(solver=solver)

    # Define investment fraction variables for stocks in each portfolio and limit sum to 1
    @variable(m, x[p in Portfolios, s in p.stocks] >= 0)
    @constraint(m, sum(x) <= 1)

    # Maximize total returns over all stocks
    @objective(m, Max, sum(p.returns[s]*x[p,s] for p in Portfolios, s in p.stocks))

    # Define stock choice indicator variables and limit number of chosen stocks to Smax and force zero investment if stock is not chosen
    @variable(m, y[s in Stocks], Bin)
    @constraint(m, sum(y) <= Smax)
    @constraint(m, [p in Portfolios, s in p.stocks], x[p,s] <= y[s])

    # Add risk constraints on Sigmahalf_p*x_p for each portfolio p
    @variable(m, Shx[p in Portfolios, s in p.stocks])
    @constraint(m, [p in Portfolios, s1 in p.stocks], Shx[p,s1] == sum(p.Sigmahalf[(s1,s2)]*x[p,s2] for s2 in p.stocks))

    for p in Portfolios
        if p.risk == :norm2
            @constraint(m, norm(Shx[p,s] for s in p.stocks) <= p.gamma)
        elseif p.risk == :robustnorm2
            lambda = @variable(m, lowerbound=0)
            @SDconstraint(m, [p.gamma [Shx[p,s] for s in p.stocks]' [x[p,s] for s in p.stocks]'; [Shx[p,s] for s in p.stocks] (diagm(fill(p.gamma, p.size)) - [lambda.*p.Delta[(s1,s2)] for s1 in p.stocks, s2 in p.stocks]) zeros(p.size, p.size); [x[p,s] for s in p.stocks] zeros(p.size, p.size) diagm(fill(lambda, p.size))] >= 0)
        # elseif p.risk == :entropy
        #     ent1 = @variable(m, [s in p.stocks])
        #     ent2 = @variable(m, [s in p.stocks])
        #     @constraint(m, sum(ent1[s] + ent2[s] for s in p.stocks) <= p.gamma^2)
        #     for s in p.stocks
        #         @Conicconstraint(m, [-ent1[s], 1 + Shx[p,s], 1] >= 0, :ExpPrimal)
        #         @Conicconstraint(m, [-ent2[s], 1 - Shx[p,s], 1] >= 0, :ExpPrimal)
        #     end
        else
            error("Invalid risk type $(p.risk)")
        end
    end

    return (m, x, y)
end


#=========================================================
Data generation functions
=========================================================#

# Generate model data from basic model options, reading portfolio data from datafiles
function generatedata(risks, counts, maxstocks, gammas, datadir, datafiles)
    N = sum(counts)
    @printf "\n\nGenerating data for %d portfolios\n" N
    @printf "\n%6s %6s %12s" "ID" "Size" "Risk type"

    Portfolios = Vector{Portfolio}(N)

    k = 0
    for b in 1:length(risks), bp in 1:counts[b] # For each risk type, each portfolio of the risk type
        k += 1

        pid = k
        pgamma = gammas[b]
        prisk = risks[b]

        (pstocks, preturns, pSigmahalf) = loadportfolio(datadir, datafiles[k], maxstocks[b]) # Read raw data for portfolio
        psize = length(pstocks)

        pDelta = Dict{Tuple{String,String},Float64}()
        if prisk == :robustnorm2
            # Generate random matrix and scale and clean zeros
            Deltahalf = randn(psize, psize)
            scalefactor = 1/10*norm([v for v in values(pSigmahalf)])/norm(vec(Deltahalf))
            @assert 1e-3 < scalefactor < 1e2
            for i in 1:psize, j in 1:psize
                val = scalefactor*Deltahalf[i,j]
                if val < 1e-3
                    Deltahalf[i,j] = 0.
                else
                    Deltahalf[i,j] = val
                end
            end

            # Fill Delta = Deltahalf*Deltahalf'
            for i in 1:psize, j in i:psize
                (s1, s2) = (pstocks[i], pstocks[j])
                pDelta[(s1,s2)] = pDelta[(s2,s1)] = vecdot(Deltahalf[i,:], Deltahalf[:,j])
            end
        end

        @printf "\n%6d %6d %12s" pid psize string(prisk)

        Portfolios[k] = Portfolio(pid, psize, pstocks, preturns, pSigmahalf, prisk, pgamma, pDelta)
    end

    stockset = Set{String}()
    for p in Portfolios
        union!(stockset, p.stocks)
    end
    Stocks = collect(stockset)

    return (Portfolios, Stocks)
end

# Load data from a .por file, returning returns, sqrt of covariance matrix, ticker names; take at most maxstocks stocks
function loadportfolio(datadir, datafile, maxstocks::Int)
    @assert endswith(datafile, ".por")
    file = open(joinpath(datadir, datafile), "r")

    n = parse(Int, chomp(readline(file)))

    if n > maxstocks
        taken = maxstocks
        takestocks = randperm(n)[1:taken]
    else
        taken = n
        takestocks = randperm(n)
    end

    rawreturns = split(chomp(readline(file)))
    @assert length(rawreturns) == n

    rawSigmahalf = zeros(n, n)
    for i in 1:n
        data = split(chomp(readline(file)))
        @assert length(data) == n
        for j in 1:n
            rawSigmahalf[i,j] = parse(Float64, data[j])
        end
    end

    @assert chomp(readline(file)) == ""

    line = chomp(readline(file))
    @assert startswith(line, "['") && endswith(line, "']")
    rawnames = split(line[3:end-2], "', '")
    @assert length(rawnames) == n

    pstocks = [String(rawnames[s]) for s in takestocks]

    preturns = Dict{String,Float64}(String(rawnames[s]) => parse(Float64, rawreturns[s]) for s in takestocks)

    pSigmahalf = Dict{Tuple{String,String},Float64}((String(rawnames[s1]), String(rawnames[s2])) => rawSigmahalf[s1,s2] for s1 in takestocks, s2 in takestocks)

    return (pstocks, preturns, pSigmahalf)
end


#=========================================================
Specify MICP solver
=========================================================#

using Pajarito

mip_solver_drives = false
log_level = 3
rel_gap = 1e-4

using CPLEX
mip_solver = CplexSolver(
    CPX_PARAM_SCRIND=(mip_solver_drives ? 1 : 0),
    # CPX_PARAM_SCRIND=1,
    CPX_PARAM_EPINT=1e-9,
    CPX_PARAM_EPRHS=1e-9,
    CPX_PARAM_EPGAP=(mip_solver_drives ? rel_gap : 0.)
)

# using SCS
# cont_solver = SCSSolver(eps=1e-3, max_iters=100000, verbose=0, warm_start=true)

# using ECOS
# cont_solver = ECOSSolver(verbose=false)

using Mosek
cont_solver = MosekSolver(LOG=0)

solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
    mip_solver=mip_solver,
    cont_solver=cont_solver,
    # solve_subp=false,
    # solve_relax=false,
    # init_sdp_soc=false,
    # sdp_soc=false,
    sdp_eig=false,
    # prim_cuts_only=true,
    # prim_cuts_always=true,
    # prim_cuts_assist=true,
    # dump_subproblems = true,
    # dump_basename = joinpath(pwd(), "portcbf/port")
)


#=========================================================
Specify model options and generate data
=========================================================#

srand(102)

risks = [:norm2, :entropy, :robustnorm2]
counts = [0, 0, 1]
maxstocks = [100, 20, 20]
gammas = [0.05, 0.05, 0.05]

datadir = joinpath(pwd(), "data")
datafiles = readdir(datadir)

(Portfolios, Stocks) = generatedata(risks, counts, maxstocks, gammas, datadir, datafiles)

Smax = round(Int, length(Stocks)/3)

@printf "\n\nChoose %d of %d unique stocks (sum of %d portfolio sizes is %d)\n\n" Smax length(Stocks) length(Portfolios) sum(p.size for p in Portfolios)


#=========================================================
Solve and print solution
=========================================================#

(m, x, y) = portfoliorisk(solver, Portfolios, Stocks, Smax)


# Save as CBF file
# name = "8,100_8,20_8,8"
# path = "/home/coey/portcbf/socexppsd/"
# ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(path, "$name.cbf"))


status = solve(m)

@printf "\nStatus = %s\n" status

@printf "\nReturns (obj) = %8.4f\n" getobjectivevalue(m)
@printf "\nTotal number chosen = %d\n" sum(round(Int, getvalue(y[s])) for s in Stocks)
@printf "\nTotal fraction invested = %8.4f\n" getvalue(sum(x))

for p in Portfolios
    @printf "\nPortfolio %d investments\n" p.id

    for s in p.stocks
        if getvalue(y[s]) > 0.1
            @printf "%6s %8.4f\n" s getvalue(x[p,s])
        end
    end
end
println()
