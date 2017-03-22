# Convex.jl test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities, Convex


name = "maximize"
x = Convex.Variable(1, :Int)
P = Convex.maximize(3x,
    x <= 10,
    x^2 <= 9
)
ConicBenchmarkUtilities.convex_to_cbf(P, name, joinpath(pwd(), "$name.cbf"))
