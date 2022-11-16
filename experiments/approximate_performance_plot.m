close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("approximate_performance_plot");

%% Parameters
rng(926); % For reproducibility purposes
N = 45000;
k = 250; % Change to generate different plots
Nts = 100;
mu = 1e-8 * N;
bandwidth = 100;
num_iter = 50;
kernel = "gaussian";
tol = 1e-9;

problems = struct();
% TODO: These datasets are having issues, fix them.
% problems.MiniBoonNE = ProblemParameters("MiniBoonNE", bandwidth, mu, rank, kernel);
% problems.a9a = ProblemParameters("a9a", bandwidth, mu, k, kernel); 

% WARNING: This problem has a very low numeric rank and requires a "large"
% tolerance when forming the Cholesky factorization for Nystrom
% preconditioners, i.e., 1e-8.
% problems.skin_nonskin = ProblemParameters("skin_nonskin", bandwidth, mu, rank, kernel);

problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, k, kernel);
problems.cadata = ProblemParameters("cadata", bandwidth, mu, k, kernel);
problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, k, kernel);
problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, k, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, k, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, k, kernel);
problems.phishing = ProblemParameters("phishing", bandwidth, mu, k, kernel);
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

%% Experiment
results = struct();
names = fieldnames(problems);
loadColors
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

    test_accuracy = @(beta) norm(Ats*beta - Yts,1) / length(Yts);
    relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
    summary = @(beta) [relres(beta) test_accuracy(beta)];

    results.(names{j}) = struct();
    [~,results.(names{j}).krill] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'spchol');
    fprintf('\tKrill iters: %d last iter error: %7.2e\n', size(results.(names{j}).krill, 1), results.(names{j}).krill(end, 1));
    [~,results.(names{j}).falkon] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'falkon');
    fprintf('\tFalkon iters: %d last iter error: %7.2e\n', size(results.(names{j}).falkon, 1), results.(names{j}).falkon(end, 1));
    [~,results.(names{j}).noprec] = approximate_krr(A_S,A_SS,mu,Ytr,summary,num_iter,tol,'');
    fprintf('\tNo preconditioner iters: %d last iter error: %7.2e\n', size(results.(names{j}).noprec, 1), results.(names{j}).noprec(end, 1));
    
    f1 = figure(2*j - 1);
    semilogy(results.(names{j}).krill(:,1), 'Color', color3)
    hold on
    semilogy(results.(names{j}).falkon(:,1), 'Color', color4)
    semilogy(results.(names{j}).noprec(:,1), 'Color', color2)
    xlabel('Iteration'); ylabel('Relative Residual')
    legend({'KRILL (Ours)','FALKON'})

    f2 = figure(2*j);
    semilogy(results.(names{j}).krill(:,2), 'Color', color3)
    hold on
    semilogy(results.(names{j}).falkon(:,2), 'Color', color5)
    semilogy(results.(names{j}).noprec(:,2), 'Color', color2)
    xlabel('Iteration'); ylabel('Test error')
    legend({'KRILL (Ours)','FALKON', "No preconditioner"})

    saveas(f1,fullfile(resultsPath, string(names{j}) +'_exact_test_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{j}) +'_exact_test_res.png'))
    saveas(f2,fullfile(resultsPath, string(names{j}) +'_exact_test_error.fig'))
    saveas(f2,fullfile(resultsPath, string(names{j}) +'_exact_test_error.png'))
end

%% Generate performance plot
loadColors
density = zeros(num_iter,3);
accuracy = 1e-6 ;
for j = 1:numel(names)
   density(min(find(results.(names{j}).krill(:,1) <= accuracy)), 1) = density(min(find(results.(names{j}).krill(:,1) <= accuracy)), 1) + 1;
   density(min(find(results.(names{j}).falkon(:,1) <= accuracy)), 2) = density(min(find(results.(names{j}).falkon(:,1) <= accuracy)), 2) + 1;
   density(min(find(results.(names{j}).noprec(:,1) <= accuracy)), 3) = density(min(find(results.(names{j}).noprec(:,1) <= accuracy)), 3) + 1;
end


cumulative = zeros(num_iter,3);
cumulative(1, :) = density(1, :);
for j = 2:num_iter
    cumulative(j, :) = density(j, :) + cumulative(j-1, :);
end

fperformance = figure();
numberproblems = numel(names);

plot(cumulative(:, 2)/numberproblems, 'Linewidth', 4, 'Color', color5) % FALKON
hold on
plot(cumulative(:, 3)/numberproblems, 'Linewidth', 4, 'Color', color2) % No Prec
plot(cumulative(:, 1)/numberproblems, 'Linewidth', 4, 'Color', color3) % KRILL
ylim([0.0 1.0])
xlabel('Iteration', 'FontSize', 24); 
ylabel('Fraction of solved problems', 'FontSize', 24)
le = legend({'FALKON', 'No preconditioner', 'KRILL (Ours)'});
set(gca,'FontSize',20)
axis([0 50 0 1])
saveas(fperformance,fullfile(resultsPath, 'performance.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'performance.png'), 'Resolution', 300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'N', 'mu', 'bandwidth', 'k' )

%% Auxiliary functions
function [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, Ntr, Nts)
    n = length(Ytr);
    idx_resh =randperm(n,n);
    Xtr = Xtr(idx_resh,:);
    Ytr = Ytr(idx_resh);

    if Ntr < size(Xtr, 1)
        Xtr = Xtr(1:Ntr,:); Ytr = Ytr(1:Ntr);
    end
    if Nts < size(Xts, 1)
        Xts = Xts(1:Nts,:); Yts = Yts(1:Nts);
    end
end

function [Xtr, Xts] = standarize(Xtr, Xts)
    X_mean = mean(Xtr); X_std = std(Xtr);
    bad_idx = find(std(Xtr) == 0);
    Xtr(:,bad_idx) = [];
    % TODO: There is a bug in the way LIBSVM loads their data, which only
    % affects a9a. This conditional is a hack to solve the issue. Implement
    % a better fix.
    if max(bad_idx) <= size(Xts, 2)
        Xts(:,bad_idx) = [];
    end
    X_mean(bad_idx) = []; X_std(bad_idx) = [];
    Xtr = (Xtr - X_mean) ./ X_std;
    Xts = (Xts - X_mean) ./ X_std;
end

function K = kernelmatrix(X1, X2, kernel, bandwidth)
    if contains(kernel, 'gaussian')
        K = exp(-pdist2(X1,X2,"euclidean").^2 / (2*bandwidth));
    elseif contains(kernel, 'laplace')
        K = exp(-pdist2(X1,X2,"minkowski",1)/bandwidth);
    else
        error('Kernel ' + kernel + ' not implemented')
    end
end