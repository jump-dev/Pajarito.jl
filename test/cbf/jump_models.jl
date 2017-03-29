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
# See examples/expdesign.jl
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
# See examples/expdesign.jl
(q, p, n, nmax) = (4, 8, 12, 3)
V = [-0.658136 0.383753 -0.601421 -0.211517 1.57874 2.03256 0.396071 -0.870703; -0.705681 1.63771 -0.304213 -0.213992 0.88695 1.54024 -0.134482 -0.0874732; -0.414197 -0.39504 1.31011 1.72996 -0.215804 -0.515882 0.15529 -0.630257; -0.375281 0.0 1.1321 -0.0720246 0.180677 0.524403 -0.220045 0.62724]
m = Model()
np = @variable(m, [j=1:p], Int, lowerbound=0, upperbound=nmax)
@constraint(m, sum(np) <= n)
t = @variable(m)
@objective(m, Max, t)
@SDconstraint(m, V * diagm(np./n) * V' - t * eye(q) >= 0)
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))


name = "sdp_cardls"
# See examples/cardls.jl
A = [-0.658136 -0.215804 -1.22825 0.636702 0.310855 0.0436465; 0.383753 -0.515882 -1.39494 -0.797658 -0.802035 1.15531; -0.601421 0.15529 0.638735 -0.16043 0.696064 -0.439435; -0.211517 -0.630257 0.614026 -1.4663 1.36299 -0.512717; 1.57874 -0.375281 -0.439124 1.75887 -0.814751 -1.56508; 2.03256 -0.003084 0.573321 -0.874149 -0.148805 0.263757; 0.396071 1.1321 -1.82076 -1.14665 -0.245664 -1.05774; -0.870703 -0.0720246 -0.343017 0.921975 -0.902467 -1.08266; -0.705681 0.180677 1.0088 0.709111 -0.269505 -1.59058; 1.63771 0.524403 0.198447 0.0235749 -1.22018 -1.69565; -0.304213 -0.220045 -0.249271 -0.0956476 -0.860636 0.119479; -0.213992 0.62724 -1.31959 0.907254 0.0394771 1.419; 0.88695 0.43794 0.440619 0.140498 -0.935278 -0.273569; 1.54024 -0.974513 -0.481017 0.41188 -0.211076 -0.618709; -0.134482 1.54252 0.850121 -0.678518 -1.20563 -2.02133; -0.0874732 0.605379 -1.06185 0.0803662 0.00117048 0.507544; -0.414197 -0.627169 -1.49419 -0.677743 0.610031 1.38788; -0.39504 0.025945 -1.36405 0.12975 -0.590624 -0.0804821; 1.31011 1.1715 3.57264 1.24484 1.78609 0.0945148; 1.72996 0.0928935 -0.351372 -1.3813 -0.903951 -0.402878]
b = [-2.4884, 0.24447, 1.25599, 1.03482, 0.56539, 2.16735, 0.274518, -0.649421, 0.288631, -0.99246, 0.91836, -0.983705, -0.408959, -0.560663, 0.00348301, -0.723511, -0.183856, 0.366346, -1.62336, -0.462939]
d = 6
s = 20
k = floor(Int, d/2)
rho = 1.
xB = 4
m = Model()
@variable(m, tau)
@variable(m, z[1:d], Bin)
@objective(m, Min, tau)
@constraint(m, sum(z) <= k)
@SDconstraint(m, [(eye(s) + 1/rho*A*diagm(z)*A') b ; b' tau] >= 0)
ConicBenchmarkUtilities.jump_to_cbf(m, name, joinpath(pwd(), "$name.cbf"))
