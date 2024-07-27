close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("approximate_eigenvalues");

%% Parameters
rng(926); % For reproducibility purposes
N = 40000;
k = 1000;
Nts = 1000;
mu = 1e-12 * N;
bandwidth = 3;
num_iter = 100;
kernel = "gaussian";
tol = 1e-9;

problems = struct();
problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, k, kernel);
problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, k, kernel);
problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, k, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, k, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, k, kernel);
problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, k, kernel);
problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, k, kernel);
problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", bandwidth, mu, k, kernel);
problems.w8a = ProblemParameters("w8a", bandwidth, mu, k, kernel);
problems.ACSIncome = ProblemParameters("ACSIncome", bandwidth, mu, k, kernel);
problems.Airlines_DepDelay_1M = ProblemParameters("Airlines_DepDelay_1M", bandwidth, mu, k, kernel);
problems.COMET_MC_SAMPLE = ProblemParameters("COMET_MC_SAMPLE", bandwidth, mu, k, kernel);
problems.creditcard = ProblemParameters("creditcard", bandwidth, mu, k, kernel);
problems.diamonds = ProblemParameters("diamonds", bandwidth, mu, k, kernel);
problems.hls4ml_lhc_jets_hlf = ProblemParameters("hls4ml_lhc_jets_hlf", bandwidth, mu, k, kernel);
problems.jannis = ProblemParameters("jannis", bandwidth, mu, k, kernel);
problems.Medical_Appointment = ProblemParameters("Medical-Appointment", bandwidth, mu, k, kernel);
problems.MNIST = ProblemParameters("MNIST", bandwidth, mu, k, kernel);
%problems.santander = ProblemParameters("santander", bandwidth, mu, k, kernel);
problems.volkert = ProblemParameters("volkert", bandwidth, mu, k, kernel);
problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, k, kernel);

%% Experiment
loadFont
loadColors

results = struct();
names = fieldnames(problems);
for j = 1:numel(names)
    fprintf('Solving %s\n',names{j});
    problem = problems.(names{j});
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    fprintf('\tOriginal training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, N, Nts);
    fprintf('\tSubsampled training size n = %d, d = %d\n\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);

    S = randsample(N, k, false);
    A_S = kernelmatrix(Xtr, Xtr(S,:), problem.Kernel, problem.Bandwidth);
    %A_SS = A_S(S,:);
    A_SS = A_S(S,:) + (N * k / mu) * eps * eye(k);
    A = A_S' * A_S + mu * A_SS;
    A = (A + A')/2;
    e = eig(A);
    e = e / max(e);
    e = flip(e);
    results.(names{j}) = struct();
    results.(names{j}).eigs = e;

    Phi = sparse_sign(2*k, N, 8);
    PhiA_S = Phi * A_S;
    H = PhiA_S' * PhiA_S + mu * A_SS;
    H = H + trace(H) * eps * eye(k);
    [V,D] = eig(H);
    d = diag(D).^-(1/2);
    A = V * (d .* (V' * A));
    A = (d' .* (A * V)) * V';
    A = (A + A')/2;
    f = eig(A);
    f = f / max(f);
    f = flip(f);
    results.(names{j}).pre_eigs = f;

    f1 = figure(j);
    subplot(1,2,1);
    plot(results.(names{j}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    xlabel('Index'); 
    ylabel('Normalized eigenvalues');
    xlim([1, k]);
    xticks([1 10 100 1000]);
    ylim([1e-14, 1]);
    yticks([1e-12, 1e-9, 1e-6, 1e-3, 1]);

    subplot(1,2,2)
    plot(results.(names{j}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    xlabel('Index'); 
    ylabel('Normalized eigenvalues');
    xlim([1, k]);
    xticks([1 10 100 1000]);
    ylim([1e-4, 1]);
    yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]);
    saveas(f1,fullfile(resultsPath, string(names{j}) +'_eigenvalues_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{j}) +'_eigenvalues_res.png'))
end

%% Generate eigenvalue plot
close all
loadFont
loadColors

feigenvalues = figure();
feigenvalues.Position = [10 10 600 300];
subplot(1,2,1);
plot(1:k, results.(names{1}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
    'MarkerSize', 4)
hold on;
for j = 1:numel(names)
    plot(1:k, results.(names{j}).eigs, '-o', 'Color', color5, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
end
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Index'); 
ylabel('Normalized eigenvalues');
xlim([1, k]);
xticks([1 10 100 1000]);
ylim([1e-14, 1]);
yticks([1e-12, 1e-9, 1e-6, 1e-3, 1]);

subplot(1,2,2);
plot(1:k, results.(names{1}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
    'MarkerSize', 4)
hold on;
for j = 1:numel(names)
    plot(1:k, results.(names{j}).pre_eigs, '-o', 'Color', color3, 'Linewidth', 0.5, ...
        'MarkerSize', 4);
end
set(gca,'xscale','log');
set(gca,'yscale','log');
xlabel('Index'); 
xlim([1, k]);
xticks([1 10 100 1000]);
ylim([1e-2, 1]);
yticks([1e-2, 1e-1, 1]);
saveas(feigenvalues,fullfile(resultsPath, 'eigenvalues.fig'))
exportgraphics(feigenvalues,fullfile(resultsPath, 'eigenvalues.png'), 'Resolution', 300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'mu', 'bandwidth', 'resultsPath' )