close all
clear all
clc
addpath("../code")
addpath("../utils")
resultsPath = createFolderForExecution("approximate_cross_validation_performance_plot");
loadColors
loadFont
%% Parameters
rng(926); % For reproducibility purposes
N = 40000;
k = 200; % Change to generate different plots (200 -- 4000)
Nts = 100; % Size of test dataset
num_iter = 100;
kernel = "gaussian";
tol = 1e-9;



%% Experiment
results = struct();
loadColors
loadFont
smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
combined_results = struct();
exponents = 1:4;
exponents_b = 1:3;
for ex_b = exponents_b
    results_b = struct();
    bandwidth = 3 * power(2, ex_b - 2);
    fprintf('Bandwidth %7.2e\n',ex_b)
    for ex = exponents
        mu = N * power(1/10, 2 * ex + 2);
        fprintf('Mu %7.2e\n',ex)
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
        problems.santander = ProblemParameters("santander", bandwidth, mu, k, kernel);
        problems.volkert = ProblemParameters("volkert", bandwidth, mu, k, kernel);
        problems.yolanda = ProblemParameters("yolanda", bandwidth, mu, k, kernel);

        names = fieldnames(problems);

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
            A_SS = A_S(S,:);
            Ats = kernelmatrix(Xts, Xtr(S,:), problem.Kernel, problem.Bandwidth);
            ASY = A_S' * Ytr;

            test_accuracy = @(beta) smape(Ats*beta, Yts);
            relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
            summary = @(beta) [relres(beta) test_accuracy(beta)];

            results.(names{j}) = struct();
            [~,results.(names{j}).krill] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'spchol');
            fprintf('\tKrill iters: %d last iter error: %7.2e\n', size(results.(names{j}).krill, 1), results.(names{j}).krill(end, 1));
            [~,results.(names{j}).falkon] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'falkon');
            fprintf('\tFalkon iters: %d last iter error: %7.2e\n', size(results.(names{j}).falkon, 1), results.(names{j}).falkon(end, 1));
            [~,results.(names{j}).noprec] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'');
            fprintf('\tNo preconditioner iters: %d last iter error: %7.2e\n', size(results.(names{j}).noprec, 1), results.(names{j}).noprec(end, 1));

        end
        results_b.("mu" + string(ex)) = results;
    end
    combined_results.("b" + string(ex_b)) = results_b;

end
%% Generate and save the plot of solved problems vs. mu
loadColors
loadFont
exponents = 1:4;
mu_values = N * power(1/10, 2*exponents+2);
zeros_ex = zeros(length(exponents),1);
num_solved = struct('krill', zeros_ex, 'falkon', zeros_ex, 'noprec', zeros_ex);
cutoff = 50;
accuracy = 1e-4;
for ex = exponents
    results = combined_results.b2.("mu" + string(ex));
    for k = 1:numel(names)
        if min(find(results.(names{k}).krill(:,1) <= accuracy)) <= cutoff
            num_solved.krill(ex) = num_solved.krill(ex) + 1;
        end
        if min(find(results.(names{k}).falkon(:,1) <= accuracy)) <= cutoff
            num_solved.falkon(ex) = num_solved.falkon(ex) + 1;
        end
        if min(find(results.(names{k}).noprec(:,1) <= accuracy)) <= cutoff
            num_solved.noprec(ex) = num_solved.noprec(ex) + 1;
        end
    end
end
f = figure;
numberproblems = numel(names);
semilogx(mu_values, num_solved.falkon/numberproblems, 'Linewidth', 4, 'Color', color1, 'LineStyle', '-.') % FALKON
hold on
plot(mu_values, num_solved.noprec/numberproblems, 'Linewidth', 4, 'Color', color5, 'LineStyle', ':') % No Prec
plot(mu_values, num_solved.krill/numberproblems, 'Linewidth', 4, 'Color', color3) % KRILL
ylim([0.0 1.0])
xlim([min(mu_values), max(mu_values)])
xlabel('Regularization coefficient'); 
ylabel('Fraction of solved problems')
le = legend({'FALKON', 'No preconditioner', 'KRILL (Ours)'}, 'Location', 'southeast');
saveas(f,fullfile(resultsPath, 'performance_mu.fig'))
exportgraphics(f,fullfile(resultsPath, 'performance_mu.png'), 'Resolution',300)
%% Generate and save the plot of solved problems vs. bandwidth
loadColors
loadFont
exponents = 1:3;
b_values = 3 * power(1/2, exponents-2);
zeros_ex = zeros(length(exponents),1);
num_solved = struct('krill', zeros_ex, 'falkon', zeros_ex, 'noprec', zeros_ex);
cutoff = 50;
accuracy = 1e-4;
for ex = exponents
    results = combined_results.("b" + string(ex)).mu3;
    for k = 1:numel(names)
        if min(find(results.(names{k}).krill(:,1) <= accuracy)) <= cutoff
            num_solved.krill(ex) = num_solved.krill(ex) + 1;
        end
        if min(find(results.(names{k}).falkon(:,1) <= accuracy)) <= cutoff
            num_solved.falkon(ex) = num_solved.falkon(ex) + 1;
        end
        if min(find(results.(names{k}).noprec(:,1) <= accuracy)) <= cutoff
            num_solved.noprec(ex) = num_solved.noprec(ex) + 1;
        end
    end
end
f = figure;
numberproblems = numel(names);
semilogx(b_values, num_solved.falkon/numberproblems, 'Linewidth', 4, 'Color', color1, 'LineStyle', '-.') % FALKON
hold on
plot(b_values, num_solved.noprec/numberproblems, 'Linewidth', 4, 'Color', color5, 'LineStyle', ':') % No Prec
plot(b_values, num_solved.krill/numberproblems, 'Linewidth', 4, 'Color', color3) % KRILL
ylim([0.0 1.0])
xlim([min(b_values), max(b_values)])
xlabel('Bandwidth'); 
ylabel('Fraction of solved problems')
le = legend({'FALKON', 'No preconditioner', 'KRILL (Ours)'}, 'Location', 'southeast');
saveas(f,fullfile(resultsPath, 'performance_bandwidth.fig'))
exportgraphics(f,fullfile(resultsPath, 'performance_bandwidth.png'), 'Resolution',300)
%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'combined_results', 'num_iter', 'N', 'Nts', 'mu', 'bandwidth', 'k', 'resultsPath' )

