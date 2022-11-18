%% Set up workspace
clear; close all; clc;
addpath('../utils')
addpath('../code')

%% Set options
implicit = false;
verbose = true;
N = 2e4;
ks = [1e2, 5e2, 1e3, 5e3]
numiters = 1000;
pcgtol = 1e-8;
choltol = 1e-9;
load_chemistry_data

%% Kernel matrix

fprintf('Building kernel matrix... ')
A = kernel(X,X);
Atest = kernel(X_test,X);
fprintf('done!\n')

test_accuracy = @(beta) norm(Atest*beta - Y_test,1) / length(Y_test);
relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Experiment
results = zeros(numiters, 2, length(ks));
for k = ks
    fprintf("Solving k %d... ", k);
    [~, result] = krr(A, mu, Y, k, [], summary, 'rpcnys', numiters, choltol, pcgtol);
    results(1:length(result(:, 1)), :, k) = result;
    fprintf('done! Number of iterations: %d\n', length(result(:, 1)));
end 


