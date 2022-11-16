close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("exact_performance_plot");

%% Parameters
% For reproducibility purposes
rng(926)
rank = 500;
N = 15000;
Nts = 100;
mu = 1e-8 * N;
bandwidth = 8;
num_iter = 250;
kernel = "gaussian";

problems = struct();
% TODO: These datasets are having issues, fix them.
% problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
% problems.MiniBoonNE = ProblemParameters("MiniBoonNE", bandwidth, mu, rank, kernel);

% WARNING: This problem has a very low numeric rank and requires a "large"
% tolerance when forming the Cholesky factorization for Nystrom
% preconditioners, i.e., 1e-8.
% problems.skin_nonskin = ProblemParameters("skin_nonskin", bandwidth, mu, rank, kernel);

problems.a9a = ProblemParameters("a9a", bandwidth, mu, rank, kernel); 
problems.cadata = ProblemParameters("cadata", bandwidth, mu, rank, kernel);
problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, rank, kernel);
problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, rank, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, rank, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, rank, kernel);
problems.phishing = ProblemParameters("phishing", bandwidth, mu, rank, kernel);
problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, rank, kernel);
problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, rank, kernel);
problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", bandwidth, mu, rank, kernel);
problems.w8a = ProblemParameters("w8a", bandwidth, mu, rank, kernel);
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
results = struct();
names = fieldnames(problems);

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
    test_accuracy = @(beta) norm(Ats*beta - Yts,1) / length(Yts);
    relres = @(beta) norm(A*beta + problem.Mu*beta - Ytr) / norm(Ytr);
    summary = @(beta) [relres(beta) test_accuracy(beta)];
    
    results.(names{k}) = struct();
    tol = 1e-9;
    [~,results.(names{k}).rpc] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter,tol);
    fprintf('\tRPC iters: %d last iter error: %7.2e\n', size(results.(names{k}).rpc, 1), results.(names{k}).rpc(end, 1));
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter,tol);
    fprintf('\tGreedy iters: %d, last iter error: %7.2e\n', size(results.(names{k}).greedy, 1), results.(names{k}).greedy(end, 1));
    [~,results.(names{k}).uniform] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter,tol);
    fprintf('\tUniform iters: %d, last iter error: %7.2e\n', size(results.(names{k}).uniform, 1), results.(names{k}).uniform(end, 1));
    [~,results.(names{k}).rls] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rlsnys',num_iter,tol);
    fprintf('\tRLS iters: %d, last iter error: %7.2e\n', size(results.(names{k}).rls, 1), results.(names{k}).rls(end, 1));
    [~,results.(names{k}).gaussian] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'gaussnys',num_iter,tol);
    fprintf('\tGaussian iters: %d, last iter error: %7.2e\n', size(results.(names{k}).gaussian, 1), results.(names{k}).gaussian(end, 1));
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter,tol);
    fprintf('\tNo precond iters: %d, last iter error: %7.2e\n\n', size(results.(names{k}).nopre, 1), results.(names{k}).nopre(end, 1));

    f1 = figure(2*k - 1);
    semilogy(results.(names{k}).rpc(:,1))
    hold on
    semilogy(results.(names{k}).gaussian(:,1))
    semilogy(results.(names{k}).greedy(:,1))
    semilogy(results.(names{k}).rls(:,1))
    semilogy(results.(names{k}).uniform(:,1))
    semilogy(results.(names{k}).nopre(:,1))
    xlabel('Iteration'); ylabel('Relative Residual')
    legend({'RPCholesky','Gaussian','Greedy', 'RLS', 'Uniform','No Preconditioner'})

    f2 = figure(2*k);
    semilogy(results.(names{k}).rpc(:,2))
    hold on
    semilogy(results.(names{k}).gaussian(:,2))
    semilogy(results.(names{k}).greedy(:,2))
    semilogy(results.(names{k}).rls(:,2))
    semilogy(results.(names{k}).uniform(:,2))
    semilogy(results.(names{k}).nopre(:,2))
    xlabel('Iteration'); ylabel('Test error')
    legend({'RPCholesky','Gaussian','Greedy', 'RLS', 'Uniform','No Preconditioner'})

    saveas(f1,fullfile(resultsPath, string(names{k}) +'_exact_test_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_exact_test_res.png'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_exact_test_error.fig'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_exact_test_error.png'))
end

%% Generate performance plot
close all
loadColors
density = zeros(num_iter,6);
accuracy = 1e-6;
for k = 1:numel(names)
   display(names{k})
   density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) = density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) + 1; 
   density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) = density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) + 1; 
   density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) = density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) + 1; 
   density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) = density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) + 1; 
   density(min(find(results.(names{k}).gaussian(:,1) <= accuracy)), 5) = density(min(find(results.(names{k}).gaussian(:,1) <= accuracy)), 5) + 1; 
   density(min(find(results.(names{k}).rls(:,1) <= accuracy)), 6) = density(min(find(results.(names{k}).rls(:,1) <= accuracy)), 6) + 1; 
end

cumulative = zeros(num_iter,6);
cumulative(1, :) = density(1, :);
for k = 2:num_iter
    cumulative(k, :) = density(k, :) + cumulative(k-1, :);
end

fperformance = figure();
numberproblems = numel(names);

% TODO: Decide whether to include Gaussian in this plot.
% plot(cumulative(:, 5)/numberproblems, 'Linewidth', 4, 'Color', 'black') % Gaussian
plot(cumulative(:, 2)/numberproblems, 'Linewidth', 4, 'Color', color1) % Greedy
hold on
plot(cumulative(:, 6)/numberproblems, 'Linewidth', 4, 'Color', color2) % RLS
plot(cumulative(:, 3)/numberproblems, 'Linewidth', 4, 'Color', color4) % Uniform
plot(cumulative(:, 4)/numberproblems, 'Linewidth', 4, 'Color', color5) % No preconditioner
plot(cumulative(:, 1)/numberproblems, 'Linewidth', 4, 'Color', color3) % RPC
ylim([0.0 1.0])
xlabel('Iteration', 'FontSize', 24); 
ylabel('Fraction of solved problems', 'FontSize', 24)
le = legend({'Greedy', 'RLS', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'northwest');
set(gca,'FontSize',20)
saveas(fperformance,fullfile(resultsPath, 'performance.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'performance.png'), 'Resolution',300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'))

%% Auxiliary functions
function [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, Ntr, Nts)
Xtr = Xtr(1:min(Ntr, end),:); Ytr = Ytr(1:min(Ntr,end));
Xts = Xts(1:min(Nts, end),:); Yts = Yts(1:min(Nts,end));
end

