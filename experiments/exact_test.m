%% Set up workspace 
clear; close all; clc;
addpath('../utils')
addpath('../code')

%% Set options
implicit = false;
verbose = true;
N = 1e4;
k = min(round(N/10),1000);
numiters = 100;

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
[~,rpcholesky] = krr(A,mu,Y,k,[],summary,'rpcnys',numiters,[],[],verbose);
[~,greedy] = krr(A,mu,Y,k,[],summary,'greedynys',numiters,[],[],verbose);
[~,unif] = krr(A,mu,Y,k,[],summary,'uninys',numiters,[],[],verbose);
[~,rlscores] = krr(A,mu,Y,k,[],summary,'rlsnys',numiters,[],[],verbose);
[~,gauss] = krr(A,mu,Y,k,[],summary,'gaussnys',numiters,[],[],verbose);
[~,noprec] = krr(A,mu,Y,[],[],summary,'',numiters,[],[],verbose);

%% Plots
close all
loadColors

f1 = figure(1);
semilogy(rpcholesky(:,1),'Color',color3)
hold on
semilogy(greedy(:,1),'--','Color',color1)
semilogy(unif(:,1),'-*','Color',color4,'MarkerIndices',...
    1:round(numiters/20):numiters,'MarkerSize',10)
semilogy(rlscores(:,1),'-.','Color',color2)
semilogy(gauss(:,1),'-s','Color',color6,'MarkerIndices',...
    1:round(numiters/20):numiters,'MarkerSize',10,'MarkerFaceColor',...
    color6)
semilogy(noprec(:,1),':','Color',color5)
xlabel('Iteration'); ylabel('Relative Residual')
legend({'RPCholesky','Greedy','Uniform','RLS','Gauss',...
    'No Preconditioner'},'location','best')

f2 = figure(2);
semilogy(rpcholesky(:,2),'Color',color3)
hold on
semilogy(greedy(:,2),'--','Color',color1)
semilogy(unif(:,2),'-*','Color',color4,'MarkerIndices',...
    1:round(numiters/20):numiters,'MarkerSize',10)
semilogy(rlscores(:,2),'-.','Color',color2)
semilogy(gauss(:,2),'-s','Color',color6,'MarkerIndices',...
    1:round(numiters/20):numiters,'MarkerSize',10,'MarkerFaceColor',...
    color6)
semilogy(noprec(:,2),':','Color',color5)
xlabel('Iteration'); ylabel('Test Error')

%% Save
resultsPath = createFolderForExecution("exact_test");
saveas(f1, fullfile(resultsPath, 'exact_test_res.fig'))
saveas(f1, fullfile(resultsPath, 'exact_test_res.png'))
saveas(f2, fullfile(resultsPath, 'exact_test_err.fig'))
saveas(f2, fullfile(resultsPath, 'exact_test_err.png'))
save(fullfile(resultsPath, 'state.mat'),'N','mu','k','implicit',...
    'bandwidth','rpcholesky','unif','noprec','greedy','gauss')
