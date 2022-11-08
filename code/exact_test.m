%% Initialize data
load('../data/homo.mat')
N = 1e4;
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
kernel = @(X1,X2) exp(-pdist2(X1,X2,"minkowski",1)/a);
A = kernel(X,X);
Atest = kernel(X_test,X);

% Stats
test_accuracy = @(beta) norm(Atest*beta - Y_test,1) / length(Y_test);
relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,rpcholesky] = krr(A,mu,Y,500,[],summary,'rpcnys');
[~,greedy] = krr(A,mu,Y,500,[],summary,'greedynys');
[~,unif] = krr(A,mu,Y,500,[],summary,'uninys');
[~,noprec] = krr(A,mu,Y,[],[],summary,'');

%% Plots
close all

figure(1)
semilogy(rpcholesky(:,1))
hold on
semilogy(greedy(:,1))
semilogy(unif(:,1))
semilogy(noprec(:,1))
xlabel('Iteration'); ylabel('Relative Residual')
legend({'RPCholesky','Greedy','Uniform','No Preconditioner'})

figure(2)
semilogy(rpcholesky(:,2))
hold on
semilogy(greedy(:,2))
semilogy(unif(:,2))
semilogy(noprec(:,2))
xlabel('Iteration'); ylabel('Test Accuracy')
legend({'RPCholesky','Greedy','Uniform','No Preconditioner'})