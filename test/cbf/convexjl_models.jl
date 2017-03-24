# Convex.jl test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities, Convex


name = "soc_optimal"
x = Convex.Variable(1, :Int)
P = Convex.minimize(-3x,
    x <= 10,
    x^2 <= 9)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "soc_infeasible"
x = Convex.Variable(1, :Int)
P = Convex.minimize(-3x,
    x >= 4,
    x^2 <= 9)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "soc_unbounded"
x = Convex.Variable(1, :Int)
t = Convex.Variable(1, :Int)
P = Convex.minimize(3x - t,
    x <= 10,
    x^2 <= 2t,
    t >= 5)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "exp_optimal"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1, Convex.Positive())
P = Convex.minimize(-3x - y,
    x >= 0,
    3x + 2y <= 10,
    exp(x) <= 10)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsoc_optimal"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1)
P = Convex.minimize(-3x - y,
    x >= 1,
    y >= 0,
    3x + 2y <= 10,
    x^2 <= 5,
    exp(y) + x <= 7)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsoc_optimal2"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1, Convex.Positive())
P = Convex.minimize(-3x - y,
    x >= 1,
    y >= -2,
    3x + 2y <= 30,
    exp(y^2) + x <= 7)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "sdpsoc_optimal"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1, Convex.Positive())
z = Convex.Semidefinite(2)
P = Convex.maximize(3x + y - z[1,1],
    x >= 0,
    3x + 2y <= 10,
    x^2 <= 4,
    z[1,2] >= 1,
    y >= z[2,2])
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "sdpsoc_unbounded"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1, Convex.Positive())
z = Convex.Semidefinite(2)
P = Convex.maximize(z[1,1] - x,
    x >= 0,
    3x + 2y <= 10,
    x^2 <= 4,
    y >= z[2,2])
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "sdpsoc_infeasible"
x = Convex.Variable(1, :Int)
y = Convex.Variable(1, Convex.Positive())
z = Convex.Semidefinite(2)
P = Convex.maximize(3x + y - z[1,1],
    x >= 2,
    3x + 2y <= 10,
    x^2 <= 4,
    z[1,2] >= 2,
    y >= z[2,2] + z[1,1])
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))


name = "expsdp_optimalD"
# D-optimal design
#   maximize    log det V*diag(lambda)*V'
#   subject to  sum(lambda)=1,  lambda >=0
(q, p, n, nmax) = (4, 8, 12, 3)
V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
np = Convex.Variable(p, :Int)
Q = Convex.Variable(q, q)
P = Convex.maximize(Convex.logdet(Q),
    Q == V * diagm(np./n) * V',
    sum(np) <= n,
    np >= 0,
    np <= nmax)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))
