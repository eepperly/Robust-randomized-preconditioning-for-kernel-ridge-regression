%% Set up workspace 
clear; close all; clc;
addpath('../utils')
addpath('../code')

%% Set options
rng('default')
implicit = false;
verbose = true;
usegauss = false;
userls = false;
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
smape = @(x,y) mean(2 * abs(x-y) ./ (abs(x)+abs(y)));
if implicit
    test_accuracy = @(beta) smape(kernmul(Atest,beta), Y_test);
    relres = @(beta) norm(kernmul(A,beta) + mu*beta - Y) / norm(Y);
else
    test_accuracy = @(beta) smape(Atest*beta, Y_test);
    relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
end
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,rpcholesky] = krr(A,mu,Y,k,[],summary,'rpcnys',numiters,[],[],verbose);
[~,greedy] = krr(A,mu,Y,k,[],summary,'greedynys',numiters,[],[],verbose);
[~,unif] = krr(A,mu,Y,k,[],summary,'uninys',numiters,[],[],verbose);
[~,noprec] = krr(A,mu,Y,[],[],summary,'',numiters,[],[],verbose);
if usegauss
    [~,gauss] = krr(A,mu,Y,k,[],summary,'gaussnys',numiters,[],[],verbose);
end
if userls
    [~,rlscores]= krr(A,mu,Y,k,[],summary,'rlsnys',numiters,[],[],verbose);
end

%% Plots
close all
loadColors
loadFont

figs = cell(2,1);
for j = 1:2
    figs{j} = figure(j);
    semilogy(noprec(:,j),':','Color',color5,'LineWidth',3)
    hold on
    semilogy(greedy(:,j),'-.','Color',color1,'LineWidth',3)
    semilogy(unif(:,j),'--','Color',color4,'LineWidth',3)
    semilogy(rpcholesky(:,j),'Color',color3,'LineWidth',3)
    if userls
        semilogy(rlscores(:,j),'-*','Color',color6,'MarkerIndices',...
            1:round(numiters/20):numiters,'MarkerSize',10)
    end
    if usegauss
        semilogy(gauss(:,j),'-s','Color',color6,'MarkerIndices',...
            1:round(numiters/20):numiters,'MarkerSize',10,'MarkerFaceColor',...
            color2)
    end
    xlabel('Iteration');
    
    if j == 1
        ylabel('Relative Residual')
        legend({'No Preconditioner','Greedy','Uniform','RPCholesky'},'location','best','FontSize',20)
    else
        ylabel('SMAPE')
    end
    set(gca,'FontSize',20)
end

%% Save
resultsPath = createFolderForExecution("exact_test");
saveas(figs{1}, fullfile(resultsPath, 'exact_test_res.fig'))
saveas(figs{1}, fullfile(resultsPath, 'exact_test_res.png'))
saveas(figs{2}, fullfile(resultsPath, 'exact_test_err.fig'))
saveas(figs{2}, fullfile(resultsPath, 'exact_test_err.png'))
save(fullfile(resultsPath, 'state.mat'),'N','mu','k','implicit',...
    'bandwidth','rpcholesky','unif','noprec','greedy')
