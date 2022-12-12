%% Initialize data
addpath('../utils')
load('../data/homo.mat')
X_test = X(N+1:min(2*N,end),:); Y_test = Y(N+1:min(2*N,end));
X = X(1:N,:); Y = Y(1:N);

% Standardization
[X,X_test] = standarize(X, X_test);

% Hyperparameters
mu = N*1.0e-8;
bandwidth = 5120;

% Kernel
kernel = @(X1,X2) kernelmatrix(X1, X2, 'laplace', bandwidth);