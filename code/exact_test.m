%% Set up workspace 
clear all; close all; clc;
addpath('../utils')

%% Set options
implicit = false;
N = 1e4;
k = min(round(N/10),1000);
numiters = 500;

load_chemistry_data

%% Kernel matrix

if implicit
    d = ones(N,1);
    A = @(S) kernel(X,X(S,:));
    Atest = @(S) kernel(X_test,X(S,:));
else
    fprintf('Building kernel matrix... ')
    A = kernel(X,X);
    Atest = kernel(X_test,X);
    fprintf('done!\n')
end

%% Stats
if implicit
    test_accuracy = @(beta) norm(kernmul(Atest,beta) - Y_test,1) ...
        / length(Y_test);
    relres = @(beta) norm(kernmul(A,beta) + mu*beta - Y) / norm(Y);
else
    test_accuracy = @(beta) norm(Atest*beta - Y_test,1) / length(Y_test);
    relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
end
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,rpcholesky] = krr(A,mu,Y,k,[],summary,'rpcnys',numiters);
[~,greedy] = krr(A,mu,Y,k,[],summary,'greedynys',numiters);
[~,unif] = krr(A,mu,Y,k,[],summary,'uninys',numiters);
[~,rlscores] = krr(A,mu,Y,k,[],summary,'rlsnys',numiters);
[~,gauss] = krr(A,mu,Y,k,[],summary,'gaussnys',numiters);
[~,noprec] = krr(A,mu,Y,[],[],summary,'',numiters);

%% Plots
close all

f1 = figure(1);
semilogy(rpcholesky(:,1))
hold on
semilogy(greedy(:,1))
semilogy(unif(:,1))
semilogy(rlscores(:,1))
semilogy(gauss(:,1))
semilogy(noprec(:,1))
xlabel('Iteration'); ylabel('Relative Residual')
legend({'RPCholesky','Greedy','Uniform','RLS','Gauss','No Preconditioner'})

f2 = figure(2);
semilogy(rpcholesky(:,2))
hold on
semilogy(greedy(:,2))
semilogy(unif(:,2))
semilogy(rlscores(:,2))
semilogy(gauss(:,2))
semilogy(noprec(:,2))
xlabel('Iteration'); ylabel('Test Error')

%% Save
resultsPath = createFolderForExecution("exact_test");
saveas(f1, fullfile(resultsPath, 'exact_test_res.fig'))
saveas(f1, fullfile(resultsPath, 'exact_test_res.png'))
saveas(f2, fullfile(resultsPath, 'exact_test_err.fig'))
saveas(f2, fullfile(resultsPath, 'exact_test_err.png'))
save(fullfile(resultsPath, 'state.mat'),'N','mu','k','implicit',...
    'bandwidth','rpcholesky','unif','noprec','greedy','gauss')
