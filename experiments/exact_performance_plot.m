close all
clear all
clc
addpath("../code") 
addpath("../utils")
resultsPath = createFolderForExecution("exact_performance_plot");

%% Parameters
% For reproducibility purposes
rng('default')
rank = 500;
N = 15000;
Nts = 4000;
mu = 1e-8 * N;
bandwidth = 10;
num_iter = 250;
kernel = "gaussian";

problems = struct();
% TODO: These datasets are having issues, fix them.
% problems.MiniBoonNE = ProblemParameters("MiniBoonNE", bandwidth, mu, rank, kernel);
% problems.a9a = ProblemParameters("a9a", bandwidth, mu, rank, kernel); 

% WARNING: This problem has a very low numeric rank and requires a "large"
% tolerance when forming the Cholesky factorization for Nystrom
% preconditioners, i.e., 1e-8.
% problems.skin_nonskin = ProblemParameters("skin_nonskin", bandwidth, mu, rank, kernel);

problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
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
<<<<<<< HEAD
    
    num_repetitions = 5;
    
    results.(names{k}).rpc = zeros(num_iter, 1);
    for j = 1:num_repetitions
        [~,rpc_instance] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter,tol, tol);
        pad_size = num_iter - length(rpc_instance);
        rpc_instance = [rpc_instance; repmat(min(rpc_instance), pad_size, 1)];
        results.(names{k}).rpc = results.(names{k}).rpc + rpc_instance/num_repetitions;
    end
    fprintf('\tRPC iters: %d last iter error average: %7.2e\n', size(results.(names{k}).rpc, 1), results.(names{k}).rpc(end, 1));
    
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter,tol, tol);
    fprintf('\tGreedy iters: %d, last iter error: %7.2e\n', size(results.(names{k}).greedy, 1), results.(names{k}).greedy(end, 1));

    results.(names{k}).uniform = zeros(num_iter, 1);
    for j = 1:num_repetitions
        [~,uniform_instance] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter,tol, tol);
        pad_size = num_iter - length(uniform_instance);
        uniform_instance = [uniform_instance; repmat(min(uniform_instance), pad_size, 1)];
        results.(names{k}).uniform = results.(names{k}).uniform + uniform_instance/num_repetitions;
    end
    fprintf('\tUniform iters: %d, last iter error average: %7.2e\n', size(results.(names{k}).uniform, 1), results.(names{k}).uniform(end, 1));
   
%     [~,results.(names{k}).rls] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rlsnys',num_iter,tol, tol);
%     fprintf('\tRLS iters: %d, last iter error: %7.2e\n', size(results.(names{k}).rls, 1), results.(names{k}).rls(end, 1));
%     
%     [~,results.(names{k}).gaussian] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'gaussnys',num_iter,tol, tol);
%     fprintf('\tGaussian iters: %d, last iter error: %7.2e\n', size(results.(names{k}).gaussian, 1), results.(names{k}).gaussian(end, 1));
%     
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter,tol, tol);
    fprintf('\tNo precond iters: %d, last iter error: %7.2e\n\n', size(results.(names{k}).nopre, 1), results.(names{k}).nopre(end, 1));

    f1 = figure(2*k - 1);
    semilogy(results.(names{k}).rpc(:,1), 'Linewidth', 4, 'Color', color3)
    hold on
%     semilogy(results.(names{k}).gaussian(:,1), 'Linewidth', 4, 'Color', color6)
    semilogy(results.(names{k}).greedy(:,1), 'Linewidth', 4, 'Color', color1)
%     semilogy(results.(names{k}).rls(:,1), 'Linewidth', 4, 'Color', color5)
    semilogy(results.(names{k}).uniform(:,1), 'Linewidth', 4, 'Color', color4)
    semilogy(results.(names{k}).nopre(:,1), 'Linewidth', 4, 'Color', color2)
    xlabel('Iteration'); ylabel('Relative Residual')
%     legend({'RPCholesky','Gaussian','Greedy', 'RLS', 'Uniform','No Preconditioner'})
    legend({'RPCholesky', 'Greedy', 'Uniform', 'No Preconditioner'})

    f2 = figure(2*k);
    semilogy(results.(names{k}).rpc(:,2), 'Linewidth', 4, 'Color', color3)
    hold on
%     semilogy(results.(names{k}).gaussian(:,2), 'Linewidth', 4, 'Color', color6)
    semilogy(results.(names{k}).greedy(:,2), 'Linewidth', 4, 'Color', color1)
%     semilogy(results.(names{k}).rls(:,2), 'Linewidth', 4, 'Color', color5)
    semilogy(results.(names{k}).uniform(:,2), 'Linewidth', 4, 'Color', color4)
    semilogy(results.(names{k}).nopre(:,2), 'Linewidth', 4, 'Color', color2)
    xlabel('Iteration'); ylabel('Test error')
%     legend({'RPCholesky','Gaussian','Greedy', 'RLS', 'Uniform','No Preconditioner'})
    legend({'RPCholesky', 'Greedy', 'Uniform', 'No Preconditioner'})
=======
    [~,results.(names{k}).rpc] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter,tol,tol);
    fprintf('\tRPC iters: %d last iter error: %7.2e\n', size(results.(names{k}).rpc, 1), results.(names{k}).rpc(end, 1));
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter,tol,tol);
    fprintf('\tGreedy iters: %d, last iter error: %7.2e\n', size(results.(names{k}).greedy, 1), results.(names{k}).greedy(end, 1));
    [~,results.(names{k}).uniform] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter,tol,tol);
    fprintf('\tUniform iters: %d, last iter error: %7.2e\n', size(results.(names{k}).uniform, 1), results.(names{k}).uniform(end, 1));
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter,tol,tol);
    fprintf('\tNo precond iters: %d, last iter error: %7.2e\n\n', size(results.(names{k}).nopre, 1), results.(names{k}).nopre(end, 1));

    f1 = figure(2*k - 1);
    semilogy(results.(names{k}).greedy(:,1), 'Color', color1, 'LineStyle', '-.')
    hold on
    semilogy(results.(names{k}).uniform(:,1), 'Color', color4, 'LineStyle', '--')
    semilogy(results.(names{k}).nopre(:,1), 'Color', color5, 'LineStyle', ':')
    semilogy(results.(names{k}).rpc(:,1), 'Color', color3)
    xlabel('Iteration'); ylabel('Relative Residual')
    legend({'Greedy', 'Uniform','No Preconditioner','RPCholesky'})

    f2 = figure(2*k);
    semilogy(results.(names{k}).greedy(:,2), 'Color', color1, 'LineStyle', '-.')
    hold on
    semilogy(results.(names{k}).uniform(:,2), 'Color', color4, 'LineStyle', '--')
    semilogy(results.(names{k}).nopre(:,2), 'Color', color5, 'LineStyle', ':')
    semilogy(results.(names{k}).rpc(:,2), 'Color', color3)
    xlabel('Iteration'); ylabel('Test error')
    legend({'Greedy', 'Uniform','No Preconditioner','RPCholesky'})
>>>>>>> main


    saveas(f1,fullfile(resultsPath, string(names{k}) +'_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_res.png'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_error.fig'))
    saveas(f2,fullfile(resultsPath, string(names{k}) +'_error.png'))
end

%% Generate performance plot
close all
loadFont
loadColors
<<<<<<< HEAD
names = fieldnames(problems);
density = zeros(num_iter,4);
accuracy = 1e-8;
=======
density = zeros(num_iter,4);
accuracy = 1e-6;
>>>>>>> main
for k = 1:numel(names)
   density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) = density(min(find(results.(names{k}).rpc(:,1) <= accuracy)), 1) + 1; 
   density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) = density(min(find(results.(names{k}).greedy(:,1) <= accuracy)), 2) + 1; 
   density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) = density(min(find(results.(names{k}).uniform(:,1) <= accuracy)), 3) + 1; 
   density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) = density(min(find(results.(names{k}).nopre(:,1) <= accuracy)), 4) + 1; 
<<<<<<< HEAD
%    density(min(find(results.(names{k}).gaussian(:,1) <= accuracy)), 5) = density(min(find(results.(names{k}).gaussian(:,1) <= accuracy)), 5) + 1; 
%    density(min(find(results.(names{k}).rls(:,1) <= accuracy)), 6) = density(min(find(results.(names{k}).rls(:,1) <= accuracy)), 6) + 1; 
=======
>>>>>>> main
end

cumulative = zeros(num_iter,4);
cumulative(1, :) = density(1, :);
for k = 2:num_iter
    cumulative(k, :) = density(k, :) + cumulative(k-1, :);
end

fperformance = figure();
numberproblems = numel(names);

<<<<<<< HEAD
% TODO: Decide whether to include Gaussian and RLS in this plot.
% plot(cumulative(:, 5)/numberproblems, 'Linewidth', 4, 'Color', 'black') % Gaussian
plot(cumulative(:, 2)/numberproblems, 'Linewidth', 4, 'Color', color1) % Greedy
hold on
% plot(cumulative(:, 6)/numberproblems, 'Linewidth', 4, 'Color', color5) % RLS
plot(cumulative(:, 3)/numberproblems, 'Linewidth', 4, 'Color', color4) % Uniform
plot(cumulative(:, 4)/numberproblems, 'Linewidth', 4, 'Color', color2) % No preconditioner
plot(cumulative(:, 1)/numberproblems, 'Linewidth', 4, 'Color', color3) % RPC
ylim([0.0 1.0])
xlabel('Iteration', 'FontSize', 24); 
ylabel('Fraction of solved problems', 'FontSize', 24)
le = legend({'Greedy', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'northwest');
set(gca,'FontSize',20)

%% Save everything
saveas(fperformance,fullfile(resultsPath, accuracy +'_performance.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'performance.png'), 'Resolution',300)
save(fullfile(resultsPath, 'state.mat'), 'problems', 'results', 'num_iter', 'N', 'mu', 'bandwidth', 'rank', 'resultsPath' )

%% Auxiliary functions
function [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, Ntr, Nts)
n = length(Ytr);
idx_resh =randperm(n,n);
nts = length(Yts);
idx_resh_ts =randperm(nts,nts);
Xtr = Xtr(idx_resh,:);
Ytr = Ytr(idx_resh);
Xts = Xts(idx_resh_ts,:);
Yts = Yts(idx_resh_ts);
Xtr = Xtr(1:min(Ntr, end),:); Ytr = Ytr(1:min(Ntr,end));
Xts = Xts(1:min(Nts, end),:); Yts = Yts(1:min(Nts,end));
end

=======
% TODO: Decide whether to include Gaussian in this plot.
plot(cumulative(:, 2)/numberproblems, 'Color', color1, 'LineStyle', '-.') % Greedy
hold on
plot(cumulative(:, 3)/numberproblems, 'Color', color4, 'LineStyle', '--') % Uniform
plot(cumulative(:, 4)/numberproblems, 'Color', color5, 'LineStyle', ':') % No preconditioner
plot(cumulative(:, 1)/numberproblems, 'Color', color3) % RPC
ylim([0.0 1.0])
xlabel('Iteration'); 
ylabel('Fraction of solved problems')
le = legend({'Greedy', 'Uniform','No Preconditioner', 'RPCholesky (Ours)'}, 'Location', 'northwest');
saveas(fperformance,fullfile(resultsPath, 'performance.fig'))
exportgraphics(fperformance,fullfile(resultsPath, 'performance.png'), 'Resolution',300)

%% Save everything
save(fullfile(resultsPath, 'state.mat'))
>>>>>>> main
