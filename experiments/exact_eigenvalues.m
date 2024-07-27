close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("exact_eigenvalues");

%% Parameters
rng('default'); % For reproducibility purposes
rank = 1000; % Change to generate different plots (500 -- 1000)
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
%problems.santander = ProblemParameters("santander", bandwidth, mu, rank, kernel);
problems.volkert = ProblemParameters("volkert", bandwidth, mu, rank, kernel);
problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, rank, kernel);

%% Experiment
loadFont
loadColors

results = struct();
names = fieldnames(problems);
for k = 1:numel(names)
    fprintf('Solving %s\n',names{k});
    problem = problems.(names{k});
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    fprintf('\tOriginal training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, N, Nts);
    fprintf('\tSubsampled training size n = %d, d = %d\n\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);

    A = kernelmatrix(Xtr, Xtr, problem.Kernel, problem.Bandwidth);
    e = eig(A) + mu;
    e = flip(e);
    results.(names{k}) = struct();
    results.(names{k}).eigs = e;

    F = rpcholesky(A, rank, min(100, ceil(rank/10)), [], diag(A));
    [U,S,~] = svd(F,'econ');
    d = (diag(S).^2 + mu).^(-1/2) - mu^(-1/2);
    B = A + mu * eye(N);
    B = U * (d .* (U' * B)) + mu^(-1/2) * B;
    B = (d' .* (B * U)) * U' + mu^(-1/2) * B;
    B = (B + B')/2;
    f = eig(B) * mu;
    f = flip(f);
    results.(names{k}).pre_eigs = f;

    f1 = figure(k);
    subplot(1,2,1);
    plot(results.(names{k}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    xlabel('Index'); 
    ylabel('Eigenvalues');
    xlim([1, N]);
    xticks([1 10 100 1000 10000]);
    yticks([1e-4 1e-2 1 1e2 1e4]);
    ylim([1/N, N]);
    yline(mu, '--', '\mu')

    subplot(1,2,2)
    plot(results.(names{k}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    xlabel('Index'); 
    ylabel('Eigenvalues');
    xlim([1, N]);
    xticks([1 10 100 1000 10000]);
    yticks([1e-4 1e-2 1 1e2 1e4]);
    ylim([1/N, N]);
    yline(mu, '--', '\mu');
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_eigenvalues_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_eigenvalues_res.png'))
end

%% Generate eigenvalue plot
close all
loadFont
loadColors

feigenvalues = figure();
feigenvalues.Position = [10 10 600 300];
subplot(1,2,1);
plot(1:N, results.(names{1}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
    'MarkerSize', 4)
hold on;
for k = 1:numel(names)
    plot(1:N, results.(names{k}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
end
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Index'); 
ylabel('Eigenvalues');
xlim([1, N]);
xticks([1 10 100 1000 10000]);
ylim([5e-4, N]);
yticks([1e-2 1 1e2 1e4]);
yline(mu, '--')

subplot(1,2,2);
plot(1:N, results.(names{1}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
    'MarkerSize', 4)
hold on;
for k = 1:numel(names)
    plot(1:N, results.(names{k}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
end
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Index'); 
xlim([1, N]);
xticks([1 10 100 1000 10000])
ylim([5e-4, N]);
yticks([1e-2 1 1e2 1e4]);
yline(mu, '--')
saveas(feigenvalues,fullfile(resultsPath, 'eigenvalues.fig'))
exportgraphics(feigenvalues,fullfile(resultsPath, 'eigenvalues.png'), 'Resolution', 300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'mu', 'bandwidth', 'rank', 'resultsPath' )