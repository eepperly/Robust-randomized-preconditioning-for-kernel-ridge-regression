%% Set up workspace 
clear; close all; clc;
addpath('../utils')
addpath('../code')

%% Set options
N = 1e4;
k = min(round(N/10),1000);
numiters = 100;

load_chemistry_data

%% Kernel
S = randsample(N,k,false);
A_S = kernel(X,X(S,:));
A_SS = A_S(S,:);
Atest = kernel(X_test,X(S,:));
ASY = A_S' * Y;

%% Stats
test_accuracy = @(beta) norm(Atest*beta - Y_test,1) / length(Y_test);
relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,stats] = approximate_krr(A_S,A_SS,mu,Y,summary,100,1e-5);

%% Plots
loadColors
close all

f1 = figure(1);
semilogy(stats(:,1),'Color',color3)
xlabel('Iteration'); ylabel('Relative Residual')

f2 = figure(2);
plot(stats(:,2),'Color',color3)
xlabel('Iteration'); ylabel('Mean Average Test Error (eV)')

%% Save
resultsPath = createFolderForExecution("approximate_test");
saveas(f1, fullfile(resultsPath, 'approximate_test_res.fig'))
saveas(f1, fullfile(resultsPath, 'approximate_test_res.png'))
saveas(f2, fullfile(resultsPath, 'approximate_test_err.fig'))
saveas(f2, fullfile(resultsPath, 'approximate_test_err.png'))
save(fullfile(resultsPath, 'state.mat'),'N','mu','k','stats','bandwidth')
