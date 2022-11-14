close all
clear all
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
bandwidth = 10;
num_iter = 250;
kernel = "gaussian";

problems = struct();
% TODO: These datasets are having issues, fix them.
% problems.HIGGS = ProblemParameters("HIGGS", bandwidth, mu, rank, kernel);
% problems.MiniBoonNE = ProblemParameters("MiniBoonNE", bandwidth, mu, rank, kernel);

problems.a9a = ProblemParameters("a9a", bandwidth, mu, rank, kernel); % name, bandwidth, mu, approximation rank, kernel
problems.cadata = ProblemParameters("cadata", bandwidth, mu, rank, kernel);
problems.cod_rna = ProblemParameters("cod-rna", bandwidth, mu, rank, kernel);
problems.connect_4 = ProblemParameters("connect-4", bandwidth, mu, rank, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", bandwidth, mu, rank, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", bandwidth, mu, rank, kernel);
problems.phishing = ProblemParameters("phishing", bandwidth, mu, rank, kernel);
problems.sensit_vehicle = ProblemParameters("sensit_vehicle", bandwidth, mu, rank, kernel);
problems.sensorless = ProblemParameters("sensorless", bandwidth, mu, rank, kernel);
problems.skin_nonskin = ProblemParameters("skin_nonskin", bandwidth, mu, rank, kernel);
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
    fprintf('\tSubsampled training size n = %d, d = %d\n', size(Xtr, 1), size(Xtr,2));
    [Xtr, Xts] = standarize(Xtr, Xts);
    
    A = kernelmatrix(Xtr, Xtr, problem.Kernel, problem.Bandwidth);
    Ats = kernelmatrix(Xts,Xtr, problem.Kernel, problem.Bandwidth);
    test_accuracy = @(beta) norm(Ats*beta - Yts,1) / length(Yts);
    relres = @(beta) norm(A*beta + problem.Mu*beta - Ytr) / norm(Ytr);
    summary = @(beta) [relres(beta) test_accuracy(beta)];
    results.(names{k}) = struct();
    [~,results.(names{k}).rpc] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter);
    fprintf('\t RPC last iter error : %7.2e\n', results.(names{k}).rpc(end, 1));
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter);
    fprintf('\t Greedy last iter error: %7.2e\n', results.(names{k}).greedy(end, 1));    
    [~,results.(names{k}).uniform] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter);
    fprintf('\t Uniform last iter error: %7.2e\n', results.(names{k}).uniform(end, 1));    
    % [~,results.(names{k}).rls] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rlsnys',num_iter);
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter);
    fprintf('\t No precond last iter error: %7.2e\n', results.(names{k}).nopre(end, 1));
    f1 = figure(k);
    semilogy(results.(names{k}).rpc(:,1))
    hold on
    semilogy(results.(names{k}).greedy(:,1))
    semilogy(results.(names{k}).uniform(:,1))
    %semilogy(rlscores(:,1))
    semilogy(results.(names{k}).nopre(:,1))
    xlabel('Iteration'); ylabel('Relative Residual')
    legend({'RPCholesky','Greedy','Uniform','No Preconditioner'})
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_exact_test_res.fig'))
    saveas(f1,fullfile(resultsPath, string(names{k}) +'_exact_test_res.png'))
end

%% Generate performance plot
close all
density = zeros(num_iter,4);
accuracy = 1e-6;
for k = 1:numel(names)
   display(names{k})
   if (names{k}) == "skin_nonskin"
       continue
   end
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
plot(cumulative(:, 1))
hold on
plot(cumulative(:, 2))
plot(cumulative(:, 3))
plot(cumulative(:, 4))
xlabel('Iteration'); ylabel('Number of solved problems')
legend({'RPCholesky','Greedy','Uniform','No Preconditioner'})
saveas(fperformance,fullfile(resultsPath, 'performance.fig'))
saveas(fperformance,fullfile(resultsPath, 'performance.png'))
%save(fullfile(resultsPath, 'state.mat'))

%% Auxiliary functions
function [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, Ntr, Nts)
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
K = exp(-pdist2(X1,X2,"euclidean").^2 / (2*bandwidth));
end
