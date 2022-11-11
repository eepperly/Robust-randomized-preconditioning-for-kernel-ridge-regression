close all
clear all
addpath("../code") 

% Gotta reproduce the results
rng(926)
rank = 800;
N = 8000;
Nts = 100;
mu = 1e-10 * N;
num_iter = 250;
kernel = "gaussian";
problems = struct();
% problems.a9a = ProblemParameters("a9a", 9, 1e-4, rank, kernel); % name, bandwidth, mu, approximation rank, kernel
problems.cadata = ProblemParameters("cadata", 5, 1e-4, rank, kernel);
problems.cod_rna = ProblemParameters("cod-rna", 5, 1e-4, rank, kernel);
problems.connect_4 = ProblemParameters("connect-4", 5, 1e-4, rank, kernel);
problems.covtype_binary = ProblemParameters("covtype.binary", 5, 1e-4, rank, kernel);
problems.ijcnn1 = ProblemParameters("ijcnn1", 5, 1e-4, rank, kernel);
problems.phishing = ProblemParameters("phishing", 5, 1e-4, rank, kernel);
problems.sensit_vehicle = ProblemParameters("sensit_vehicle", 5, 1e-4, rank, kernel);
problems.sensorless = ProblemParameters("sensorless", 5, 1e-4, rank, kernel);
problems.skin_nonskin = ProblemParameters("skin_nonskin", 5, 1e-4, rank, kernel);
problems.YearPredictionMSD = ProblemParameters("YearPredictionMSD", 5, 1e-4, rank, kernel);
problems.w8a = ProblemParameters("w8a", 5, 1e-4, rank, kernel);
problems.ACSIncome = ProblemParameters("ACSIncome", 5, 1e-4, rank, kernel);
problems.Airlines_DepDelay_1M = ProblemParameters("Airlines_DepDelay_1M", 5, 1e-4, rank, kernel);
% Saved with single precision ignoring from now 
% problems.COMET_MC_SAMPLE = ProblemParameters("COMET_MC_SAMPLE", 5, 1e-4, rank, kernel);
% problems.HIGGS = ProblemParameters("HIGGS", 5, 1e-4, rank, kernel);
problems.creditcard = ProblemParameters("creditcard", 5, 1e-4, rank, kernel);
problems.diamonds = ProblemParameters("diamonds", 5, 1e-4, rank, kernel);
problems.hls4ml_lhc_jets_hlf = ProblemParameters("hls4ml_lhc_jets_hlf", 5, 1e-4, rank, kernel);
problems.jannis = ProblemParameters("jannis", 5, 1e-4, rank, kernel);
problems.Medical_Appointment = ProblemParameters("Medical-Appointment", 5, 1e-4, rank, kernel);
% problems.MiniBoonNE = ProblemParameters("MiniBoonNE", 5, 1e-4, rank, kernel);
problems.MNIST = ProblemParameters("MNIST", 5, 1e-4, rank, kernel);
problems.santander = ProblemParameters("santander", 5, 1e-4, rank, kernel);
problems.volkert = ProblemParameters("volkert", 5, 1e-4, rank, kernel);
problems.yolanda = ProblemParameters("yolanda", 5, 1e-4, rank, kernel);

results = struct();
names = fieldnames(problems);

for k = 1:numel(names)
    fprintf('Solving %s\n',names{k})
    try
    problem = problems.(names{k});
    [Xtr, Ytr, Xts, Yts] = problem.loaddata();
    [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, N, Nts);
    [Xtr, Xts] = standarize(Xtr, Xts);
    
    A = kernelmatrix(Xtr, Xtr, problem.Kernel, problem.Bandwidth);
    Ats = kernelmatrix(Xts,Xtr, problem.Kernel, problem.Bandwidth);
    test_accuracy = @(beta) norm(Ats*beta - Yts,1) / length(Yts);
    relres = @(beta) norm(A*beta + problem.Mu*beta - Ytr) / norm(Ytr);
    summary = @(beta) [relres(beta) test_accuracy(beta)];
    results.(names{k}) = struct();
    [~,results.(names{k}).rpc] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rpcnys',num_iter);
    fprintf('\t RPC error last iter: %7.2e\n', results.(names{k}).rpc(end, 1));
    [~,results.(names{k}).greedy] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'greedynys',num_iter);
    fprintf('\t Greedy error last iter: %7.2e\n', results.(names{k}).greedy(end, 1));    
    [~,results.(names{k}).uniform] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'uninys',num_iter);
    fprintf('\t Uniform error last iter: %7.2e\n', results.(names{k}).uniform(end, 1));    
    % [~,results.(names{k}).rls] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'rlsnys',num_iter);
    [~,results.(names{k}).nopre] = krr(A,problem.Mu,Ytr,problem.ApproximationRank,[],summary,'',num_iter);
    fprintf('\t No pre error last iter: %7.2e\n', results.(names{k}).nopre(end, 1));
    f1 = figure(k);
    semilogy(results.(names{k}).rpc(:,1))
    hold on
    semilogy(results.(names{k}).greedy(:,1))
    semilogy(results.(names{k}).uniform(:,1))
    %semilogy(rlscores(:,1))
    semilogy(results.(names{k}).nopre(:,1))
    xlabel('Iteration'); ylabel('Relative Residual')
    legend({'RPCholesky','Greedy','Uniform','No Preconditioner'})
    saveas(f1,'../results/'+ string(names{k}) +'_exact_test_res.fig')
    catch
        warning('There was an error processing: ' + string(names{k}));
    end
end

%% Generate performance plot
close all
num_iter = 300;
density = zeros(num_iter,4);
accuracy = 1e-8;
for k = 1:numel(names)
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
saveas(fperformance,'../results/'+ string(accuracy) +'_performance.fig')

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
Xtr(:,bad_idx) = []; Xts(:,bad_idx) = [];
X_mean(bad_idx) = []; X_std(bad_idx) = [];
Xtr = (Xtr - X_mean) ./ X_std;
Xts = (Xts - X_mean) ./ X_std;
end

function K = kernelmatrix(X1, X2, kernel, bandwidth)
K = exp(-pdist2(X1,X2,"euclidean").^2 / (2*bandwidth));
end
