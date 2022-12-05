%% Set up workspace
clear; close all; clc;
addpath('../utils')
addpath('../code')
resultsPath = createFolderForExecution("convergence_vs_rank");

%% Set options
verbose = true;
N = 64e3;
ks = [1e2 2e2 5e2 1e3 2e3 4e3, 8e3, 16e3, 32e3];
numiters = 1000;
pcgtol = 1e-4;
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
ks = [1e2 2e2 5e2 1e3 2e3 5e3, 1e4, 2e4]
results = zeros(numiters, 2, length(ks));
final_iter = zeros(length(ks));
counter = 1;
for k = ks
    fprintf("Solving k %d... ", k);
    [~, result] = krr(A, mu, Y, k, [], summary, 'rpcnys', numiters, choltol, pcgtol);
    results(1:length(result(:, 1)), :, k) = result;
    fprintf('done! Number of iterations: %d\n', length(result(:, 1)));
    final_iter(counter) = length(result(:, 1));
    counter = counter + 1;
end 

%% Plot 
fperformance = figure();
loadColors
semilogx(ks, final_iter, 'Linewidth', 4, 'Color', color3)
xlabel('Number of centers'); ylabel('Iterations to convergence');
set(gca, 'FontSize', 22);
%% Save everything
saveas(fperformance,fullfile(resultsPath, 'convergence_vs_rank.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'convergence_vs_rank.png'), 'Resolution',300)
save(fullfile(resultsPath, 'state.mat'), 'results', 'N', 'mu', 'bandwidth', 'ks', 'final_iter')