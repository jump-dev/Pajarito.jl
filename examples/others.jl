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




% JoÃ«lle Skaf - 04/24/08
% (a figure is generated)
%
% Suppose y \in\reals^n is a Gaussian random variable with zero mean and
% covariance matrix R = \Expect(yy^T), with sparse inverse S = R^{-1}
% (S_ij = 0 means that y_i and y_j are conditionally independent).
% We want to estimate the covariance matrix R based on N independent
% samples y1,...,yN drawn from the distribution, and using prior knowledge
% that S is sparse
% A good heuristic for estimating R is to solve the problem
%           maximize    logdet(S) - tr(SY) - lambda*sum(sum(abs(S)))
%           subject to  S >= 0
% where Y is the sample covariance of y1,...,yN, and lambda is a sparsity
% parameter to be chosen or tuned.
% A figure showing the sparsity (number of nonzeros) of S versus lambda
% is generated.

% Input data
randn('state',0);
n = 10;
N = 100;
Strue = sprandsym(n,0.5,0.01,1);
nnz_true = sum(Strue(:)>1e-4);
R = inv(full(Strue));
y_sample = sqrtm(R)*randn(n,N);
Y = cov(y_sample');
Nlambda = 20;
lambda = logspace(-2, 3, Nlambda);
nnz = zeros(1,Nlambda);

for i=1:Nlambda
    disp(['i = ' num2str(i) ', lambda(i) = ' num2str(lambda(i))]);
    % Maximum likelihood estimate of R^{-1}
    cvx_begin sdp quiet
        variable S(n,n) symmetric
        maximize log_det(S) - trace(S*Y) - lambda(i)*sum(sum(abs(S)))
        S >= 0
    cvx_end
    nnz(i) = sum(S(:)>1e-4);
end

figure;
semilogx(lambda, nnz);
hold on;
semilogx(lambda, nnz_true*ones(1,Nlambda),'r');
xlabel('\lambda');
legend('nonzeros in S', 'nonzeros in R^{-1}');
