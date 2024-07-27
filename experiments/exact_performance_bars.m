close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("exact_performance_bars");

%% Parameters
rng('default'); % For reproducibility purposes
rank = 1000;
N = 15000;
Nts = 10;
mu = 1e-7 * N;  % Change to generate different plots (1e-10 -- 1e-7)
bandwidth = 3;
num_iter = 250;
kernel = "gaussian";
trials = 100;

problems = struct();
problems.COMET_MC_SAMPLE = ProblemParameters("COMET_MC_SAMPLE", bandwidth, mu, rank, kernel);
% problems.creditcard = ProblemParameters("creditcard", bandwidth, mu, rank, kernel);
% problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
% problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, rank, kernel);
% problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, rank, kernel);
% problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, rank, kernel);
% problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, rank, kernel);
% problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, rank, kernel);
problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, rank, kernel);
% problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", bandwidth, mu, rank, kernel);
% problems.w8a = ProblemParameters("w8a", bandwidth, mu, rank, kernel);
% problems.ACSIncome = ProblemParameters("ACSIncome", bandwidth, mu, rank, kernel);
% problems.Airlines_DepDelay_1M = ProblemParameters("Airlines_DepDelay_1M", bandwidth, mu, rank, kernel);
% problems.diamonds = ProblemParameters("diamonds", bandwidth, mu, rank, kernel);
% problems.hls4ml_lhc_jets_hlf = ProblemParameters("hls4ml_lhc_jets_hlf", bandwidth, mu, rank, kernel);
% problems.jannis = ProblemParameters("jannis", bandwidth, mu, rank, kernel);
% problems.Medical_Appointment = ProblemParameters("Medical-Appointment", bandwidth, mu, rank, kernel);
% problems.MNIST = ProblemParameters("MNIST", bandwidth, mu, rank, kernel);
% problems.santander = ProblemParameters("santander", bandwidth, mu, rank, kernel);
% problems.volkert = ProblemParameters("volkert", bandwidth, mu, rank, kernel);
% problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, rank, kernel);

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
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter,tol,tol);
    fprintf('\tGreedy iters: %d, last iter error: %7.2e\n', size(results.(names{k}).greedy, 1), results.(names{k}).greedy(end, 1));
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter,tol,tol);
    fprintf('\tNo precond iters: %d, last iter error: %7.2e\n\n', size(results.(names{k}).nopre, 1), results.(names{k}).nopre(end, 1));

    for trial = 1:trials
        fprintf('\t\tTrial %d\n', trial);
        [~,results.(names{k}).rpc_many(:,trial)]...
            = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],...
            relres,'rpcnys',num_iter,0,0);
        [~,results.(names{k}).uniform_many(:,trial)]...
            = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],...
            relres,'uninys',num_iter,0,0);
        [~,results.(names{k}).rls_many(:,trial)]...
            = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],...
            relres,'rlsnys',num_iter,0,0);
        [~,results.(names{k}).rff_many(:,trial)]...
            = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],...
            relres,'rff',num_iter,0,0,[],Xtr,problem.Bandwidth);
    end
    f1 = figure(k);
    semilogy(results.(names{k}).greedy(:,1), 'Color', color1, 'LineStyle', '-.')
    hold on
    semilogy(median(results.(names{k}).uniform_many,2), 'Color', color4, 'LineStyle', '--')
    semilogy(median(results.(names{k}).rls_many,2), 'Color', color6,'LineStyle', '-')
    semilogy(median(results.(names{k}).rff_many,2), 'Color', color8,'LineStyle', '--')
    semilogy(results.(names{k}).nopre(:,1), 'Color', color5, 'LineStyle', ':')
    semilogy(median(results.(names{k}).rpc_many,2), 'Color', color3)
    set(gca, 'YScale', 'log')
    xlabel('Iteration'); ylabel('Relative residual')
    if strcmp(names{k}, 'COMET_MC_SAMPLE')
        axis([0 250 1e-10 1e2])
    else
        axis([0 250 1e-5 1e1])
    end
    le = legend({'Greedy','Uniform','RLS','RFF','No Preconditioner', 'RPCholesky (Ours)'}, ...
        'Location', 'southwest', 'AutoUpdate','off');
    plot_shaded(1:num_iter,...
        quantile(results.(names{k}).uniform_many,0.2,2),...
        quantile(results.(names{k}).uniform_many,0.8,2),...
        color4, 'Linewidth', 4,'LineStyle', '--')
    plot_shaded(1:num_iter,...
        quantile(results.(names{k}).rls_many,0.2,2),...
        quantile(results.(names{k}).rls_many,0.8,2),...
        color6, 'Linewidth', 4)
    plot_shaded(1:num_iter,...
        quantile(results.(names{k}).rff_many,0.2,2),...
        quantile(results.(names{k}).rff_many,0.8,2),...
        color8, 'Linewidth', 4, 'LineStyle','--')
    plot_shaded(1:num_iter,...
        quantile(results.(names{k}).rpc_many,0.2,2),...
        quantile(results.(names{k}).rpc_many,0.8,2),...
        color3, 'Linewidth', 4)
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_bars.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_bars.png'))
end

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'mu', 'bandwidth', 'rank', 'resultsPath' )

%% Check the size of quantiles
k = 1;
accuracy = 1e-3;
[~, idx] = max(results.(names{k}).rpc_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).rls_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).uniform_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).rff_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
k = 2;
accuracy = 1e-3;
[~, idx] = max(results.(names{k}).rpc_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).rls_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).uniform_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{k}).rff_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
