addpath('../utils')
addpath('../code')
N = 4e4;
k = 500;
[X,bandwidth] = smile(N);
mu = 1e-7 * N;
tol = 1e-9;
num_iter = 100;
A = kernelmatrix(X, X, "gaussian", bandwidth);
y = X(:,1).^2/3 + X(:,2) .* sin(X(:,1));
relres = @(beta) norm(A*beta + mu*beta - y) / norm(y);
[~,nopre] = krr(A,mu,y,k,[],relres,'',num_iter,tol,tol,true);
[~,rpc] = krr(A,mu,y,k,[],relres,'rpcnys',num_iter,tol,tol,true);
[~,rls] = krr(A,mu,y,k,[],relres,'rlsnys',num_iter,tol,tol,true);

figure
semilogy(nopre)
hold on
semilogy(rpc)
semilogy(rls)
legend({'No preconditioner', 'RPC', 'RLS'},'Location','southwest')