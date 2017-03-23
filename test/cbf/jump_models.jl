# JuMP test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities, JuMP


name = "soc_infeas_bin"
# Hijazi example - no feasible binary points in the ball centered at 1/2
dim = 5
m = Model()
@variable(m, x[1:dim], Bin)
@variable(m, t)
@constraint(m, t == sqrt(dim-1)/2)
@constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
@objective(m, Min, 0)
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))
