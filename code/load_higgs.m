%% Initialize data
addpath('../utils')
load('../data/preprocessed/HIGGS.mat')
X = double(Xtr);  X_test = double(Xts);
Y = double(Ytr)'; Y_test = double(Yts)';
N = size(X,1);

% Standardization
[X,X_test] = standarize(X, X_test);

% Hyperparameters
mu = N*1.0e-8;
bandwidth = 20;

% Kernel
kernel = @(X1,X2) kernelmatrix(X1, X2, 'gaussian', bandwidth);