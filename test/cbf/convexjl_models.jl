# jl test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities, Convex


name = "soc_optimal"
x = Variable(1, :Int)
P = minimize(-3x,
    x <= 10,
    x^2 <= 9)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "soc_infeasible"
x = Variable(1, :Int)
P = minimize(-3x,
    x >= 4,
    x^2 <= 9)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "exp_optimal"
x = Variable(1, :Int)
y = Variable(1, Positive())
P = minimize(-3x - y,
    x >= 0,
    3x + 2y <= 10,
    exp(x) <= 10)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsoc_optimal"
x = Variable(1, :Int)
y = Variable(1)
P = minimize(-3x - y,
    x >= 1,
    y >= 0,
    3x + 2y <= 10,
    x^2 <= 5,
    exp(y) + x <= 7)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsoc_optimal2"
x = Variable(1, :Int)
y = Variable(1, Positive())
P = minimize(-3x - y,
    x >= 1,
    y >= -2,
    3x + 2y <= 30,
    exp(y^2) + x <= 7)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsoc_optimal3"
x = Variable(1, :Int)
y = Variable(1, Positive())
P = minimize(-3x - y,
    x + exp((y+1)^2 - 2x) <= 3)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "sdpsoc_optimal"
x = Variable(1, :Int)
y = Variable(1, Positive())
z = Semidefinite(2)
P = maximize(3x + y - z[1,1],
    x >= 0,
    3x + 2y <= 10,
    x^2 <= 4,
    z[1,2] >= 1,
    y >= z[2,2])
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "sdpsoc_infeasible"
x = Variable(1, :Int)
y = Variable(1, Positive())
z = Semidefinite(2)
P = maximize(3x + y - z[1,1],
    x >= 2,
    3x + 2y <= 10,
    x^2 <= 4,
    z[1,2] >= 2,
    y >= z[2,2] + z[1,1])
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsdp_optimalD"
# See examples/expdesign.jl
(q, p, n, nmax) = (4, 8, 12, 3)
V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
np = Variable(p, :Int)
Q = Variable(q, q)
P = maximize(logdet(Q),
    Q == V * diagm(np./n) * V',
    sum(np) <= n,
    np >= 0,
    np <= nmax)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "exp_gatesizing"
# See examples/gatesizing.jl
yUB = 3
fe = [1, 0.8, 1, 0.7, 0.7, 0.5, 0.5] .* [1, 2, 1, 1.5, 1.5, 1, 2]
Cout6 = 10
Cout7 = 10
y = Variable(7)
z = Variable(yUB, 7, Positive(), :Bin)
D1 = exp(-y[1]) + exp(-y[1] + y[4])
D2 = 2 * exp(-y[2]) + exp(-y[2] + y[4]) + exp(-y[2] + y[5])
D3 = 2 * exp(-y[3]) + exp(-y[3] + y[5]) + exp(-y[3] + y[7])
D4 = 2 * exp(-y[4]) + exp(-y[4] + y[6]) + exp(-y[4] + y[7])
D5 = exp(-y[5]) + exp(-y[5] + y[7])
D6 = Cout6 * exp(-y[6])
D7 = Cout7 * exp(-y[7])
P = minimize(maximum([
    (D1+D4+D6), (D1+D4+D7), (D2+D4+D6), (D2+D4+D7), (D2+D5+D7), (D3+D5+D6), (D3+D7)
    ]),
    y >= 0,
    sum(fe .* exp(y)) <= 20,
    sum(exp(y)) <= 100)
for i in 1:7
    P.constraints += (sum(z[:,i]) == 1)
    P.constraints += (y[i] == sum([log(j) * z[j,i] for j=1:yUB]))
end
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))
