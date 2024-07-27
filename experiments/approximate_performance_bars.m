close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("approximate_performance_bars");

%% Parameters
rng(926); % For reproducibility purposes
N = 40000;
k = 1000; % Change to generate different plots (200 -- 4000)
Nts = 1000; % Size of test dataset
mu = 1e-6 * N;
bandwidth = 3;
num_iter = 100;
kernel = "gaussian";
tol = 1e-9;
trials = 100;

problems = struct();
% problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, k, kernel);
% problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, k, kernel);
% problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, k, kernel);
% problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, k, kernel);
% problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, k, kernel);
% problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, k, kernel);
% problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, k, kernel);
% problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", bandwidth, mu, k, kernel);
% problems.w8a = ProblemParameters("w8a", bandwidth, mu, k, kernel);
% problems.ACSIncome = ProblemParameters("ACSIncome", bandwidth, mu, k, kernel);
% problems.Airlines_DepDelay_1M = ProblemParameters("Airlines_DepDelay_1M", bandwidth, mu, k, kernel);
% problems.COMET_MC_SAMPLE = ProblemParameters("COMET_MC_SAMPLE", bandwidth, mu, k, kernel);
problems.creditcard = ProblemParameters("creditcard", bandwidth, mu, k, kernel);
% problems.diamonds = ProblemParameters("diamonds", bandwidth, mu, k, kernel);
% problems.hls4ml_lhc_jets_hlf = ProblemParameters("hls4ml_lhc_jets_hlf", bandwidth, mu, k, kernel);
% problems.jannis = ProblemParameters("jannis", bandwidth, mu, k, kernel);
% problems.Medical_Appointment = ProblemParameters("Medical-Appointment", bandwidth, mu, k, kernel);
% problems.MNIST = ProblemParameters("MNIST", bandwidth, mu, k, kernel);
% problems.santander = ProblemParameters("santander", bandwidth, mu, k, kernel);
% problems.volkert = ProblemParameters("volkert", bandwidth, mu, k, kernel);
problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, k, kernel);

%% Experiment
results = struct();
names = fieldnames(problems);
loadColors
loadFont
smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
for j = 1:numel(names)
    fprintf('Solving %s\n',names{j})
    problem = problems.(names{j});
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    fprintf('\tOriginal training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    n = min(size(Xtr, 1), N);
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, n, Nts);
    fprintf('\tSubsampled training size n = %d, d = %d\n\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);

    S = randsample(n, k, false);
    A_S = kernelmatrix(Xtr, Xtr(S,:), problem.Kernel, problem.Bandwidth);
    %A_SS = A_S(S,:);
    A_SS = A_S(S,:) + (N * k / mu) * eps * eye(k);
    Ats = kernelmatrix(Xts, Xtr(S,:), problem.Kernel, problem.Bandwidth);
    ASY = A_S' * Ytr;

    test_accuracy = @(beta) smape(Ats*beta, Yts);
    relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
    summary = @(beta) [relres(beta) test_accuracy(beta)];

    results.(names{j}) = struct();
    results.(names{j}).falkon_many = zeros(num_iter,trials);
    results.(names{j}).krill_many = zeros(num_iter,trials);
    results.(names{j}).noprec_many = zeros(num_iter,trials);
    fprintf('\n\tMaking confidence interval plots\n')
    for trial = 1:trials
        fprintf('\t\tTrial %d\n', trial);
        S = randsample(n, k, false);
        A_S = kernelmatrix(Xtr, Xtr(S,:), problem.Kernel,...
            problem.Bandwidth);
        A_SS = A_S(S,:);
        Ats = kernelmatrix(Xts, Xtr(S,:), problem.Kernel,...
            problem.Bandwidth);
        ASY = A_S' * Ytr;
        relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
        [~,results.(names{j}).falkon_many(:,trial)]...
            = approximate_krr(A_S,A_SS,mu,Ytr,relres,num_iter,0,'falkon');
        [~,results.(names{j}).krill_many(:,trial)]...
            = approximate_krr(A_S,A_SS,mu,Ytr,relres,num_iter,0,'spchol');
        [~,results.(names{j}).noprec_many(:,trial)]...
            = approximate_krr(A_S,A_SS,mu,Ytr,relres,num_iter,0,'');
    end
    f1 = figure(j);
    semilogy(median(results.(names{j}).falkon_many,2), 'Color', color1, 'LineStyle', '--')
    hold on
    semilogy(median(results.(names{j}).noprec_many,2), 'Color', color5, 'LineStyle', '-.')
    semilogy(median(results.(names{j}).krill_many,2), 'Color', color3,'LineStyle', '-')
    legend({'FALKON', 'No preconditioner', 'KRILL (Ours)'}, ...
        'AutoUpdate', 'off', 'Location', 'southwest')
    plot_shaded(1:num_iter,...
        quantile(results.(names{j}).falkon_many,0.2,2),...
        quantile(results.(names{j}).falkon_many,0.8,2),...
        color1, 'Linewidth', 4, 'LineStyle', '-.')
    plot_shaded(1:num_iter,...
        quantile(results.(names{j}).noprec_many,0.2,2),...
        quantile(results.(names{j}).noprec_many,0.8,2),...
        color5, 'Linewidth', 4,'LineStyle', ':')
    plot_shaded(1:num_iter,...
        quantile(results.(names{j}).krill_many,0.2,2),...
        quantile(results.(names{j}).krill_many,0.8,2),...
        color3, 'Linewidth', 4)
    set(gca, 'YScale', 'log')
    xlabel('Iteration'); ylabel('Relative residual')
    axis([0 100 1e-10 1e0])
    saveas(f1,fullfile(resultsPath, string(names{j}) +'_bars.fig'))
    saveas(f1,fullfile(resultsPath, string(names{j}) +'_bars.png'))
end

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'Nts', 'mu', 'bandwidth', 'k', 'resultsPath' )

%% Check the size of quantiles
j = 1;
accuracy = 1e-4;
[~, idx] = max(results.(names{j}).krill_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{j}).noprec_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{j}).falkon_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
j = 2;
accuracy = 1e-4;
[~, idx] = max(results.(names{j}).krill_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{j}).noprec_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
[~, idx] = max(results.(names{j}).falkon_many <= accuracy);
(quantile(idx,.8) - quantile(idx,.2)) / median(idx)
