# JuMP test models, converted to CBF for conic model tests

using ConicBenchmarkUtilities, JuMP


name = "soc_infeasible2"
# Hijazi example - no feasible binary points in the ball centered at 1/2
dim = 5
m = Model()
@variable(m, x[1:dim], Bin)
@variable(m, t)
@constraint(m, t == sqrt(dim-1)/2)
@constraint(m, norm(x[j]-0.5 for j in 1:dim) <= t)
@objective(m, Min, 0)
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))


name = "sdp_optimalA"
# A-optimal design
#   minimize    Trace (sum_i lambdai*vi*vi')^{-1}
#   subject to  lambda >= 0, 1'*lambda = 1
(q, p, n, nmax) = (4, 8, 12, 3)
V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
m = Model()
np = @variable(m, [j=1:p], Int, lowerbound=0, upperbound=nmax)
@constraint(m, sum(np) <= n)
u = @variable(m, [i=1:q], lowerbound=0)
@objective(m, Min, sum(u))
E = eye(q)
for i=1:q
    @SDconstraint(m, [V * diagm(np./n) * V' E[:,i]; E[i,:]' u[i]] >= 0)
end
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))


name = "sdp_optimalE"
# E-optimal design
#   maximize    w
#   subject to  sum_i lambda_i*vi*vi' >= w*I
#               lambda >= 0,  1'*lambda = 1;
(q, p, n, nmax) = (4, 8, 12, 3)
V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
m = Model()
np = @variable(m, [j=1:p], Int, lowerbound=0, upperbound=nmax)
@constraint(m, sum(np) <= n)
t = @variable(m)
@objective(m, Max, t)
@SDconstraint(m, V * diagm(np./n) * V' - t * eye(q) >= 0)
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))
