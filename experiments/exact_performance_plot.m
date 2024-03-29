close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("exact_performance_plot");

%% Parameters
rng('default'); % For reproducibility purposes
rank = 500; % Change to generate different plots (500 -- 1000)
N = 15000;
Nts = 10;
mu = 1e-7 * N; 
bandwidth = 3;
num_iter = 250;
kernel = "gaussian";

problems = struct();
problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, rank, kernel);
problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, rank, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, rank, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, rank, kernel);
problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, rank, kernel);
problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, rank, kernel);
problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", bandwidth, mu, rank, kernel);
problems.w8a = ProblemParameters("w8a", bandwidth, mu, rank, kernel);
problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
problems.ACSIncome = ProblemParameters("ACSIncome", bandwidth, mu, rank, kernel);
problems.Airlines_DepDelay_1M = ProblemParameters("Airlines_DepDelay_1M", bandwidth, mu, rank, kernel);
problems.COMET_MC_SAMPLE = ProblemParameters("COMET_MC_SAMPLE", bandwidth, mu, rank, kernel);
problems.creditcard = ProblemParameters("creditcard", bandwidth, mu, rank, kernel);
problems.diamonds = ProblemParameters("diamonds", bandwidth, mu, rank, kernel);
problems.hls4ml_lhc_jets_hlf = ProblemParameters("hls4ml_lhc_jets_hlf", bandwidth, mu, rank, kernel);
problems.jannis = ProblemParameters("jannis", bandwidth, mu, rank, kernel);
problems.Medical_Appointment = ProblemParameters("Medical-Appointment", bandwidth, mu, rank, kernel);
problems.MNIST = ProblemParameters("MNIST", bandwidth, mu, rank, kernel);
problems.santander = ProblemParameters("santander", bandwidth, mu, rank, kernel);
problems.volkert = ProblemParameters("volkert", bandwidth, mu, rank, kernel);
problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, rank, kernel);

%% Experiment
loadFont
loadColors
results = struct();
names = fieldnames(problems);
loadColors
smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
for k = 1:numel(names)
    fprintf('Solving %s\n',names{k})
    problem = problems.(names{k});
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    fprintf('\tOriginal training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, N, Nts);
    fprintf('\tSubsampled training size n = %d, d = %d\n\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);
    
    A = kernelmatrix(Xtr, Xtr, problem.Kernel, problem.Bandwidth);
    Ats = kernelmatrix(Xts, Xtr, problem.Kernel, problem.Bandwidth);
    test_accuracy = @(beta) smape(Ats*beta, Yts);
    relres = @(beta) norm(A*beta + problem.Mu*beta - Ytr) / norm(Ytr);
    summary = @(beta) [relres(beta) test_accuracy(beta)];
    
    results.(names{k}) = struct();
    tol = 1e-9;
    [~,results.(names{k}).rpc] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter,tol,tol);
    fprintf('\tRPC iters: %d last iter error: %7.2e\n', size(results.(names{k}).rpc, 1), results.(names{k}).rpc(end, 1));
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter,tol,tol);
    fprintf('\tGreedy iters: %d, last iter error: %7.2e\n', size(results.(names{k}).greedy, 1), results.(names{k}).greedy(end, 1));
    [~,results.(names{k}).uniform] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter,tol,tol);
    fprintf('\tUniform iters: %d, last iter error: %7.2e\n', size(results.(names{k}).uniform, 1), results.(names{k}).uniform(end, 1));
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter,tol,tol);
    fprintf('\tNo precond iters: %d, last iter error: %7.2e\n\n', size(results.(names{k}).nopre, 1), results.(names{k}).nopre(end, 1));

    f2 = figure(2*k);
    semilogy(results.(names{k}).greedy(:,2), 'Color', color1, 'LineStyle', '-.')
    hold on
    semilogy(results.(names{k}).uniform(:,2), 'Color', color4, 'LineStyle', '--')
    semilogy(results.(names{k}).nopre(:,2), 'Color', color5, 'LineStyle', ':')
    semilogy(results.(names{k}).rpc(:,2), 'Color', color3)
    xlabel('Iteration'); ylabel('Test error')

    f1 = figure(2*k - 1);
    semilogy(results.(names{k}).greedy(:,1), 'Color', color1, 'LineStyle', '-.')
    hold on
    semilogy(results.(names{k}).uniform(:,1), 'Color', color4, 'LineStyle', '--')
    semilogy(results.(names{k}).nopre(:,1), 'Color', color5, 'LineStyle', ':')
    semilogy(results.(names{k}).rpc(:,1), 'Color', color3)
    xlabel('Iteration'); ylabel('Relative Residual')
    %legend({'Greedy', 'Uniform','No Preconditioner','RPCholesky'})

    saveas(f1,fullfile(resultsPath, string(names{k}) +'_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_res.png'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_error.fig'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_error.png'))
end

%% Generate performance plot
close all
loadFont
loadColors
density = zeros(num_iter,4);
names = fieldnames(problems);
accuracy = 1e-3;
for k = 1:numel(names)
   density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) = density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) + 1; 
   density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) = density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) + 1; 
   density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) = density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) + 1; 
   density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) = density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) + 1; 
end

cumulative = zeros(num_iter,4);
cumulative(1, :) = density(1, :);
for k = 2:num_iter
    cumulative(k, :) = density(k, :) + cumulative(k-1, :);
end

fperformance = figure();
numberproblems = numel(names);
plot(cumulative(:, 2)/numberproblems, 'Color', color1, 'LineStyle', '-.') % Greedy
hold on
plot(cumulative(:, 3)/numberproblems, 'Color', color4, 'LineStyle', '--') % Uniform
plot(cumulative(:, 4)/numberproblems, 'Color', color5, 'LineStyle', ':') % No preconditioner
plot(cumulative(:, 1)/numberproblems, 'Color', color3) % RPC
ylim([0.0 1.0])
xlabel('Iteration'); 
ylabel('Fraction of solved problems')
%le = legend({'Greedy', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'northwest');
saveas(fperformance,fullfile(resultsPath, 'performance.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'performance.png'), 'Resolution',300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'mu', 'bandwidth', 'rank', 'resultsPath' )
