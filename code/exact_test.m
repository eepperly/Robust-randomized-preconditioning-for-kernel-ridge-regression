%% Initialize data
load('../data/homo.mat')
N = 1e3;
X_test = X(N+1:(2*N),:); Y_test = Y(N+1:(2*N));
X = X(1:N,:); Y = Y(1:N);

% Standardization
X_mean = mean(X); X_std = std(X);
bad_idx = find(std(X) == 0);
X(:,bad_idx) = []; X_test(:,bad_idx) = [];
X_mean(bad_idx) = []; X_std(bad_idx) = [];
X = (X - X_mean) ./ X_std;
X_test = (X_test - X_mean) ./ X_std;

% Hyperparameters
mu = N*1.0e-8;
a = 5120;

% Kernel matrix
A = exp(-pdist2(X,X,"minkowski",1)/a);

% Stats
test_eval = @(beta) kernel(X_test,X) * beta;
test_accuracy = @(beta) norm(test_eval(beta) - Y_test,1) / length(Y_test);
relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,rpcholesky] = krr(A,mu,Y,100,[],summary,'rpcnys');
[~,noprec] = krr(A,mu,Y,100,[],summary,'');
semilogy(rpcholesky(:,1)); hold on; semilogy(noprec(:,1))
xlabel('Iteration'); ylabel('Relative Residual')
legend({'RPCholesky','No Preconditioner'})