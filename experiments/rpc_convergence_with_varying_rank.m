%% Set up workspace
clear; close all; clc;
addpath('../utils')
addpath('../code')
resultsPath = createFolderForExecution("convergence_vs_rank");
rng('default')

%% Set options
verbose = true;
N = 1e5;
ks = [1e2 2e2 5e2 1e3 2e3 5e3, 1e4, 2e4];
numiters = 1000;
pcgtol = 1e-3;
choltol = 1e-9;
implicit = true;
load_chemistry_data

%% Kernel matrix

if implicit
    d = ones(N,1);
    A = @(S) kernel(X,X(S,:));
else
    fprintf('Building kernel matrix... ')
    A = kernel(X,X);
    fprintf('done!\n')
end

%% Stats
smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
if implicit
    relres = @(beta) norm(kernmul(A,beta) + mu*beta - Y) / norm(Y);
else
    relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
end
summary = @(beta) relres(beta);

%% Experiment
results = zeros(numiters, length(ks));
final_iter = zeros(1, length(ks));
counter = 1;
for k = ks
    fprintf("Solving k %d... ", k);
    [~, result] = krr(A, mu, Y, k, [], summary, 'rpcnys', numiters, choltol, pcgtol, true);
    results(1:length(result), counter) = result;
    fprintf('done! Number of iterations: %d\n', length(result));
    final_iter(counter) = length(result);
    counter = counter + 1;
end

%% Experiment - uniform
results_uniform = zeros(numiters, length(ks));
final_iter_uniform = zeros(1, length(ks));
counter = 1;
for k = ks
    fprintf("Solving k %d... ", k);
    [~, result] = krr(A, mu, Y, k, [], summary, 'uninys', numiters, choltol, pcgtol, true);
    results_uniform(1:length(result), counter) = result;
    fprintf('done! Number of iterations: %d\n', length(result));
    final_iter_uniform(counter) = length(result);
    counter = counter + 1;
end

%% Experiment - greedy
results_greedy = zeros(numiters, length(ks));
final_iter_greedy = zeros(1, length(ks));
counter = 1;
for k = ks
    fprintf("Solving k %d... ", k);
    [~, result] = krr(A, mu, Y, k, [], summary, 'greedynys', numiters, choltol, pcgtol, true);
    results_greedy(1:length(result), counter) = result;
    fprintf('done! Number of iterations: %d\n', length(result));
    final_iter_greedy(counter) = length(result);
    counter = counter + 1;
end

%% Plot 
fperformance = figure();
loadColors
loadFont
semilogx(ks, final_iter_greedy, '-.', 'LineWidth', 3,'Color',color1)
hold on
semilogx(ks, final_iter_uniform, '--', 'LineWidth', 4, 'Color', color4)
semilogx(ks, final_iter, 'Linewidth', 4, 'Color', color3)
hold off
xlabel('Approximation rank $r$'); ylabel('Iterations to convergence');
set(gca, 'FontSize', 22);
legend({'Greedy', 'Uniform','RPCholesky'},'location','best','FontSize',20)

%% Save everything
saveas(fperformance,fullfile(resultsPath, 'convergence_vs_rank.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'convergence_vs_rank.png'), 'Resolution',300)
save(fullfile(resultsPath, 'state.mat'), 'results', 'results_uniform', ...
    'results_greedy', 'final_iter_greedy', 'final_iter_uniform', ...
    'final_iter', 'N', 'mu', 'bandwidth', 'ks')
