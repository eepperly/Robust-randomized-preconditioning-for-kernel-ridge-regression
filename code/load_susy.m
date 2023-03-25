%% Initialize data
addpath('../utils')
load('../data/preprocessed/SUSY.mat')
N = 5e5; Ntest = 5e5;
X = A(1:N,:);  X_test = A(N+1:Ntest,:);
Y = b(1:N,:);  Y_test = b(N+1:Ntest,:);
N = size(X,1);

% Standardization
[X,X_test] = standarize(X, X_test);

% Hyperparameters
mu = N*2e-8;
bandwidth = 4;

% Kernel
kernel = @(X1,X2) kernelmatrix(X1, X2, 'gaussian', bandwidth);