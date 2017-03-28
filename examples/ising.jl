# Ising model for jointly estimating the covariane parameters and graphical structure of binary-valued Markov random fields
# Using L_0, L_1, or L_2 norm regularization to induce sparsity

using Convex, Pajarito

# Generate random matrix of true underlying Markov random field structure (F), and parameter matrix (C)
# n is number of nodes in the graph
function gen_true(n)

    return (F, C)
end

# Generate a list of random samples using the true underlying Markov random field (given by F, C)
# m is the number of samples to generate
function gen_samples(n, F, C, m)


    return S
end

# Set up Convex.jl model, solve, print solution and compare to true model structure
function ising(n, F, C, m, S, Lk, lam, Yb, degb)
    X = Array{Variable,2}(n, n)
    Y = Array{Variable,2}(n, n)
    for i in 1:n, j in (i+1):n
        X[i,j] = Variable(:Bin)
        Y[i,j] = Variable()
    end

    energy = logsumexp([-sum(Y[i,j] * S[k,i] * S[k,j] for i in 1:n, j in (i+1):n) for s in 1:m])

    if Lk == 0
        regularizer = sum(X[i,j] for i in 1:n, j in (i+1):n)
    elseif Lk == 1
        regularizer = sum(abs(Y[i,j]) for i in 1:n, j in (i+1):n)
    elseif Lk == 2
        regularizer = norm(Y[i,j] for i in 1:n, j in (i+1):n)
    end

    P = minimize(energy + lam * regularizer)

    if isfinite(Yb)
        for i in 1:n, j in (i+1):n
            P.constraints += (Y[i,j] >= -Yb * Xij)
            P.constraints += (Y[i,j] <=  Yb * Xij)
        end
    else
        for i in 1:n, j in (i+1):n
            P.constraints += (X[i,j] >= quadoverlin(Y[i,j], Variable(Positive())))
        end
    end

    if isfinite(degb)
        for i in 1:n
            P.constraints += (sum(X[i,j] for j in (i+1):n) + sum(X[j,i] for j in 1:(i-1)) <= degb)
        end
    end

    # @show conic_problem(P)
    solve!(P, solver)

    println("Estimated graph structure (X) and parameters (Y) compared to true structure (F) and parameters (C):")
    @printf "\n%3s %3s | %3s %3s | %9s %9s" " i " " j " "Fij" "Xij" "Cij" "Yij"
    for i in 1:n, j in (i+1):n
        if (round(Int, (X[i,j]).value) > 0) || (round(Int, F[i,j]) > 0)
            @printf "\n%3d %3d | %3d %3d | %9.3e %9.3e" i j round(Int, F[i,j]) round(Int, (Y[i,j]).value) C[i,j] (Y[i,j]).value
        end
    end
    @printf "\nEnergy of solution = %9.3e" evaluate(energy)
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


# using SCS
# cont_solver = SCSSolver(eps=1e-5, max_iters=1000000, verbose=0)

using ECOS
cont_solver = ECOSSolver(verbose=false)


solver = PajaritoSolver(
    mip_solver_drives=mip_solver_drives,
    log_level=log_level,
    rel_gap=rel_gap,
	mip_solver=mip_solver,
	cont_solver=cont_solver,
    init_exp=true,
)


#=========================================================
Generate data
=========================================================#

# Number of nodes in graph
n = 10

# Number of samples
m = 30



gen data info ....

gen_true(n)
gen_samples(n)


#=========================================================
Specify model info
=========================================================#

# Type of objective regularization: Lk gives the type of norm
# L0 (cardinality), L1 (lasso), L2 (ridge)
Lk = 0

# Objective regularization parameter
if Lk == 0
    lambda = 1.
elseif Lk == 1
    lambda = 1.
elseif Lk == 2
    lambda = 1.
end

# Bound on absolute value of parameters in graph, used for big-M constraints
# If infinite, use an "unbounded variables formulation" to set Yij = 0 when Xij = 0, introducing SOCs
yB = 5.

# Upper bound on number of edges from each node that can be in the estimated structure
degb = round(Int, n/3)


#=========================================================
Solve Convex.jl model
=========================================================#

ising(n, F, C, m, S, Lk, lam, Yb, degb)










####
nSamp = 1000

lam = 0.1
thUB = 0.3
maxDeg = 4

iNode = 1

srand(100)

fileSamp = pwd() * "/isingData/ising9_03.csv"
fileOrig = pwd() * "/isingData/adja_9.csv"

# ####
println("reading original node-node matrix")
allOrig = readcsv(ascii(fileOrig), Int8)
(lenOrig, widthOrig) = size(allOrig)
@assert lenOrig == widthOrig "data is not a square matrix"

println("reading sample node data")
@time allSamp = readcsv(ascii(fileSamp), Int8)
(nData, nNode) = size(allSamp)
@assert nNode == lenOrig "number of nodes does not match original arc data"

println("number of nodes = $nNode")
println("number of sample vectors = $nData")

indSamps = shuffle(collect(1:nData))[1:nSamp]
samps = allSamp[indSamps,:]

####
# nSamp = 2
#
# lam = 0.1
# thUB = 0.3
# maxDeg = 1
#
# iNode = 1
#
# nNode = 3
# nData = 2
#
# samps = [1 -1 -1; -1 1 -1]













###data 9, 10
c = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
b = [0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,0.0]

I = [32,41,1,4,7,10,13,16,19,22,25,28,33,42,1,4,7,10,13,16,19,22,25,28,34,43,1,4,7,10,13,16,19,22,25,28,35,44,1,4,7,10,13,16,19,22,25,28,36,45,1,4,7,10,13,16,19,22,25,28,37,46,1,4,7,10,13,16,19,22,25,28,38,47,1,4,7,10,13,16,19,22,25,28,39,48,1,4,7,10,13,16,19,22,25,28,40,49,3,31,6,31,9,31,12,31,15,31,18,31,21,31,24,31,27,31,30,31,31,32,41,50,51,33,42,50,34,43,50,35,44,50,36,45,50,37,46,50,38,47,50,39,48,50,40,49,50]
J = [1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,21,21,21,21,22,22,22,23,23,23,24,24,24,25,25,25,26,26,26,27,27,27,28,28,28,29,29,29]
V = [-1.0,1.0,-1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,1.0,1.0,-1.0,1.0,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,-1.0,-0.1,1.0,-0.3,-0.3,1.0,-1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0]
A = sparse(I, J, V, length(b), length(c))

cone_con = [(:ExpPrimal,1:3),(:ExpPrimal,4:6),(:ExpPrimal,7:9),(:ExpPrimal,10:12),(:ExpPrimal,13:15),(:ExpPrimal,16:18),(:ExpPrimal,19:21),(:ExpPrimal,22:24),(:ExpPrimal,25:27),(:ExpPrimal,28:30),(:Zero,31:31),(:NonNeg,32:40),(:NonNeg,41:49),(:NonNeg,50:50),(:Zero,51:51)]
cone_var = [(:Free,1:29)]

types_var = [:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Bin,:Bin,:Bin,:Bin,:Bin,:Bin,:Bin,:Bin,:Bin]


####data 3, 2
# c = [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
# b = [0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]
#
# I = [8,11,1,4,9,12,1,4,10,13,3,7,6,7,7,8,11,14,15,9,12,14,10,13,14]
# J = [1,1,2,2,2,2,3,3,3,3,4,4,5,5,6,7,7,7,7,8,8,8,9,9,9]
# V = [-1.0,1.0,-1.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,-0.5,-1.0,-0.5,1.0,-0.3,-0.3,1.0,-1.0,-0.3,-0.3,1.0,-0.3,-0.3,1.0]
# A = sparse(I, J, V, length(b), length(c))
#
# cone_con = [(:ExpPrimal,1:3),(:ExpPrimal,4:6),(:Zero,7:7),(:NonNeg,8:10),(:NonNeg,11:13),(:NonNeg,14:14),(:Zero,15:15)]
# cone_var = [(:Free,1:9)]
#
# types_var = [:Cont,:Cont,:Cont,:Cont,:Cont,:Cont,:Bin,:Bin,:Bin]



using Pajarito, MathProgBase
using Gurobi
using ECOS
# using SCS

# const conic_solver = SCSSolver(
#     max_iters = 10000,
#     eps = 1e-6
# )

const conic_solver = ECOSSolver(
    verbose = 0,
    )

# m = MathProgBase.ConicModel(conic_solver)
# MathProgBase.loadproblem!(m, c, A, b, cone_con, cone_var)
# MathProgBase.optimize!(m)
#
# @show MathProgBase.getobjval(m)
# @show MathProgBase.getsolution(m)
# @show MathProgBase.getdual(m)

const pajarito_solver = PajaritoSolver(
    solver_mip = GurobiSolver(OutputFlag=1,InfUnbdInfo=1),
    solver_conic = conic_solver)

m = MathProgBase.ConicModel(pajarito_solver)
MathProgBase.loadproblem!(m, c, A, b, cone_con, cone_var)
MathProgBase.setvartype!(m, types_var)
MathProgBase.optimize!(m)

@show MathProgBase.getobjval(m)
@show MathProgBase.getsolution(m)
