%% Set up workspace 
clear all; close all; clc;
addpath('../utils')

%% Set options
rng(910);
implicit = false;
N = 1e5;
r = min(round(N/10),1000);
numiters = 100;

load_chemistry_data

%% Kernel matrix
fprintf('Building kernel matrix... ')
A = kernel(X,X);
Atest = kernel(X_test,X);
fprintf('done!\n')


%% Stats
test_error = @(beta) 2/length(Y_test)*sum(abs(Atest*beta - Y_test)./(abs(Atest*beta)+abs(Y_test)));
relres = @(beta) norm(A*beta + mu*beta - Y) / norm(Y);
summary = @(beta) [relres(beta) test_error(beta)];

%% Run with RPCholesky
fprintf('Solving KRR probem...')
[~,rpcholesky] = krr(A,mu,Y,r,[],summary,'rpcnys',numiters);
fprintf('done!\n')
clear A Atest
%% Restricted KRR
k = [0.01*N,0.025*N,0.05*N,0.075*N,0.1*N,0.2*N,0.3*N,0.4*N,0.5*N]; %Number of centers
TestErr_RKRR = zeros(9,1);
for i = 1:9
    fprintf('Building kernel matrix... ')
    S = randsample(N,k(i),false);
    A_S = kernel(X,X(S,:));
    A_SS = A_S(S,:);
    Atest = kernel(X_test,X(S,:));
    ASY = A_S' * Y;
    fprintf('done!\n')
    
    test_accuracy = @(beta) 2/length(Y_test)*sum(abs(Atest*beta - Y_test)./(abs(Atest*beta)+abs(Y_test)));
    
    fprintf('Solving approximate KRR probem...')
    alpha = (A_S'*A_S + mu*A_SS+k(i)*eps*eye(k(i))) \ (A_S'*Y);
    fprintf('done!\n')
    
    TestErr_RKRR(i) = test_accuracy(alpha);
    close all
end
clear S A_S A_SS Atest ASY


%% Subsampled KRR
n = [0.01*N,0.025*N,0.05*N,0.075*N,0.1*N,0.2*N,0.3*N,0.4*N,0.5*N]; %Subsample size
TestErr_SSKRR = zeros(9,1);
for i = 1:9
    fprintf('Building kernel matrix... ')
    S = randsample(N,n(i),false);
    A_SS = kernel(X(S,:),X(S,:));
    Atest = kernel(X_test,X(S,:));
    b = Y(S);
    fprintf('done!\n')
    
    test_accuracy = @(beta) 2/length(Y_test)*sum(abs(Atest*beta - Y_test)./(abs(Atest*beta)+abs(Y_test)));
    
    fprintf('Solving subsampled KRR probem...')
    alpha = (A_SS + mu*eye(n(i))) \ b;
    fprintf('done!\n')
    
    TestErr_SSKRR(i) = test_accuracy(alpha);
    close all
end
clear A_SS Atest b

%% Plotting 

loadColors
loadFont 
f1 = figure(1);
loglog(k,TestErr_RKRR,'LineStyle',':','LineWidth',3,'Color',color5)
hold on 
loglog(k,min(rpcholesky(:,2))*ones(9,1),'LineStyle','--','LineWidth',3,'Color',color3)
xlim([k(1) k(end)]) 
ylim([0.0250 TestErr_RKRR(1)])
xlabel('Number of centers'); ylabel('SMAPE')
legend({'Restricted', 'Exact'},'FontSize',20,...
        'Location','northeast')
set(gca,'FontSize',20)

f2 = figure(2);
loglog(k,TestErr_SSKRR,'LineStyle','-.','LineWidth',3,'Color',color2)
hold on 
loglog(k,min(rpcholesky(:,2))*ones(9,1),'LineStyle','--','LineWidth',3,'Color',color3)
xlim([k(1) k(end)]) 
ylim([0.0250 TestErr_SSKRR(1)])
xlabel('Number of subsampled datapoints'); ylabel('SMAPE')
legend({'Subsampled', 'Exact'},'FontSize',20,...
        'Location','northeast')
set(gca,'FontSize',20)

f3 = figure(3);
loglog(k,TestErr_SSKRR,'LineStyle','-.','LineWidth',3,'Color',color2)
hold on 
loglog(k,TestErr_RKRR,'LineStyle',':','LineWidth',3,'Color',color5)
hold on
loglog(k,min(rpcholesky(:,2))*ones(9,1),'LineStyle','--','LineWidth',3,'Color',color3)
xlim([k(1) k(end)]) 
ylim([0.0250 TestErr_SSKRR(1)])
xlabel('Number of centers/subsampled datapoints'); ylabel('SMAPE')
legend({'Restricted', 'Subsampled', 'Exact'},'FontSize',20,...
        'Location','northeast')
set(gca,'FontSize',20)

resultsPath = createFolderForExecution("homo100k");
saveas(f1, fullfile(resultsPath, 'homo100k_RKRRtesterr.fig'))
saveas(f1, fullfile(resultsPath, 'homo100k_RKRRtesterr.png'))
saveas(f2, fullfile(resultsPath, 'homo100k_SSKRRtesterr.fig'))
saveas(f2, fullfile(resultsPath, 'homo100k_SSKRRtesterr.png'))
saveas(f3, fullfile(resultsPath, 'homo100k_TestErr.fig'))
saveas(f3, fullfile(resultsPath, 'homo100k_TestErr.png'))
save(fullfile(resultsPath, 'homo100k_test.mat'),'rpcholesky','k','TestErr_RKRR','TestErr_SSKRR')
