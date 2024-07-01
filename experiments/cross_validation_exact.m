close all
clear all
clc
addpath("../code")
addpath("../utils")
resultsPath = createFolderForExecution("cross_validation_performance_plot");
loadColors
loadFont
%% Parameters
rng('default'); % For reproducibility purposes
rank = 500; % Change to generate different plots (500 -- 1000)
N = 10000;
Nts = 10;
num_iter = 100;
kernel = "gaussian";


%% Experiment
combined_results = struct();
exponents = 1:4;
exponents_b = 1:3;
for ex_b = exponents_b
    results_b = struct();
    bandwidth = 3 * power(2, ex_b - 2);
    fprintf('Bandwidth %7.2e\n',ex_b)
    for ex = exponents
        mu = N * power(1/10, 2 * ex);
        fprintf('Mu %7.2e\n',ex)
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

        loadFont
        loadColors
        loadColors
        smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
        results = struct();
        num_solved = struct('rpc', 0, 'greedy', 0, 'uniform', 0, 'nopre', 0);
        names = fieldnames(problems);
        for k = 1:numel(names)
            [tempResults, tempNumSolved] = runSingleExperiment(problems.(names{k}), N, Nts, num_iter, 1e-9);
            results.(names{k}) = tempResults;
            num_solved.rpc = num_solved.rpc + tempNumSolved.rpc;
            num_solved.greedy = num_solved.greedy + tempNumSolved.greedy;
            num_solved.uniform = num_solved.uniform + tempNumSolved.uniform;
            num_solved.nopre = num_solved.nopre + tempNumSolved.nopre;
        end
        results_b.("mu" + string(ex)) = results;
    end
    combined_results.("b" + string(ex_b)) = results_b;
end
%% Generate and save the plot of solved problems vs. mu
exponents = 1:4;
mu_values = N * power(1/10, 2*exponents);
zeros_ex = zeros(length(exponents),1);
num_solved = struct('rpc', zeros_ex, 'greedy', zeros_ex, 'uniform', zeros_ex, 'nopre', zeros_ex);
cutoff = 50;
accuracy = 1e-4;
for ex = exponents
    results = combined_results.b2.("mu" + string(ex));
    for k = 1:numel(names)
        if min(find(results.(names{k}).rpc(:,1) <= accuracy)) <= cutoff
            num_solved.rpc(ex) = num_solved.rpc(ex) + 1;
        end
        if min(find(results.(names{k}).greedy(:,1) <= accuracy)) <= cutoff
            num_solved.greedy(ex) = num_solved.greedy(ex) + 1;
        end
        if min(find(results.(names{k}).uniform(:,1) <= accuracy)) <= cutoff
            num_solved.uniform(ex) = num_solved.uniform(ex) + 1;
        end
        if min(find(results.(names{k}).nopre(:,1) <= accuracy)) <= cutoff
            num_solved.nopre(ex) = num_solved.nopre(ex) + 1;
        end
    end
end
f = figure;
numberproblems = numel(names);
semilogx(mu_values, num_solved.greedy/numberproblems, 'Color', color1, 'LineStyle', '-.') % Greedy
hold on
plot(mu_values, num_solved.uniform/numberproblems, 'Color', color4, 'LineStyle', '--') % Uniform
plot(mu_values, num_solved.nopre/numberproblems, 'Color', color5, 'LineStyle', ':') % No preconditioner
plot(mu_values, num_solved.rpc/numberproblems, 'Color', color3) % RPC
ylim([0.0 1.0])
xlim([min(mu_values) max(mu_values)])
xlabel('Regularization coefficient'); 
ylabel('Fraction of solved problems')
le = legend({'Greedy', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'southeast');
saveas(f,fullfile(resultsPath, 'performance_mu.fig'))
exportgraphics(f,fullfile(resultsPath, 'performance_mu.png'), 'Resolution',300)

%% Generate and save the plot of solved problems vs. mu
exponents = 1:3;
b_values = 3 * power(1/2, exponents - 2);
zeros_ex = zeros(length(exponents),1);
num_solved = struct('rpc', zeros_ex, 'greedy', zeros_ex, 'uniform', zeros_ex, 'nopre', zeros_ex);
cutoff = 50;
accuracy = 1e-4;
for ex = exponents
    results = combined_results.("b" +string(ex)).mu4;
    for k = 1:numel(names)
        if min(find(results.(names{k}).rpc(:,1) <= accuracy)) <= cutoff
            num_solved.rpc(ex) = num_solved.rpc(ex) + 1;
        end
        if min(find(results.(names{k}).greedy(:,1) <= accuracy)) <= cutoff
            num_solved.greedy(ex) = num_solved.greedy(ex) + 1;
        end
        if min(find(results.(names{k}).uniform(:,1) <= accuracy)) <= cutoff
            num_solved.uniform(ex) = num_solved.uniform(ex) + 1;
        end
        if min(find(results.(names{k}).nopre(:,1) <= accuracy)) <= cutoff
            num_solved.nopre(ex) = num_solved.nopre(ex) + 1;
        end
    end
end
f = figure;
numberproblems = numel(names);
semilogx(b_values, num_solved.greedy/numberproblems, 'Color', color1, 'LineStyle', '-.') % Greedy
hold on
plot(b_values, num_solved.uniform/numberproblems, 'Color', color4, 'LineStyle', '--') % Uniform
plot(b_values, num_solved.nopre/numberproblems, 'Color', color5, 'LineStyle', ':') % No preconditioner
plot(b_values, num_solved.rpc/numberproblems, 'Color', color3) % RPC
ylim([0.0 1.0])
xlabel('Bandwidth'); 
ylabel('Fraction of solved problems')
le = legend({'Greedy', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'northwest');
saveas(f,fullfile(resultsPath, 'performance_bandwidth.fig'))
exportgraphics(f,fullfile(resultsPath, 'performance_bandwidth.png'), 'Resolution',300)
%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'combined_results', 'num_iter', 'N', 'mu', 'bandwidth', 'rank', 'resultsPath');