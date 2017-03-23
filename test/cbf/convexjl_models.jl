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
