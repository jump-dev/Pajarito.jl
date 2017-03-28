cardls

function solveModel(m, n, A, b, k, rho, int_solver)
    mod = Model(solver=int_solver)
    @defVar(mod, tau)
    @defVar(mod, z[j=1:n], Bin)
    @setObjective(mod, Min, tau)
    @addConstraint(mod, sum(z) <= k)
    @addSDPConstraint(mod, [(eye(m) + 1/rho * A * diagm(z) * A') b ; b' tau] >= 0)
    solve(mod)
    return getValue(X)
end


# computing restricted isometry constants, from gally16

function solveModel(n, A, k, int_solver)
    AtA = A'*A
    mod = Model(solver=int_solver)
    @defVar(mod, z[i=1:n], Bin)
    @defVar(mod, X[i=1:n,j=1:n], SDP)
    @setObjective(mod, Min, sum{dot(AtA[k,:], X[:,k]), k=1:n}) #Tr(A'AX)
    @addConstraint(mod, sum{X[k,k], k=1:n} == 1)
    @addConstraint(mod, pxz[i=1:n,j=1:n], X[i,j] <= z[j])
    @addConstraint(mod, nxz[i=1:n,j=1:n], -X[i,j] <= z[j])
    @addConstraint(mod, sum(z) <= k)
    solve(mod)
    return (getValue(z), getValue(X))
end



# computing truss topology from gally16

using JuMP, Pajarito

function solveModel(m, n, Es, As, l, Q, A, cHat, int_solver)
    mod = Model(solver=int_solver)
    @defVar(mod, tau <= cHat)
    @defVar(mod, X[e in Es, a in As], Bin)
    @setObjective(mod, Min, sum{l[e] * a * X[e,a], e in Es, a in As})
    @addConstraint(mod, sum{X[e,a], a in As} <= 1)
    @addSDPConstraint(mod, [2 * tau * eye(n) Q' ; Q sum{A[e] * a * X[e,a], e in Es, a in As}] >= 0)
    solve(mod)
    return getValue(X)
end
