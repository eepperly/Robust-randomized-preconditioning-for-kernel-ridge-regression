%% Set options
addpath('../utils')
rng(926); %Set random seed
%% Problem setup
load('SUSY.mat')
N = 4500000; 
Xtr0 = A(1:N,:); Y = b(1:N); %First 4,500,000 points for training set
Atst = A(N+1:end,:); Y_test = b(N+1:end); %Last 500,000 points for test set
clear A b


[Xtr,Xtst] = standarize(Xtr0,Atst); %Standardize the data matrices

Y(Y==0) = -1; Y_test(Y_test==0) = -1; %Convert labels from 0,1 to -1,+1

mu = 10^(-3); %regularization 
bandwidth = 16; %%sigma^2
kernel = @(X1,X2) kernelmatrix(X1, X2, 'gaussian', bandwidth); %define kernel 
k = 10000;  S = randsample(N,k,false); %sample 10,000 centers

clear Xtr0 Atst

%% Building kernel matrix
fprintf('Building kernel matrix... ')
    A_S = kernel(Xtr,Xtr(S,:)); A_SS = A_S(S,:);
    Atest = kernel(Xtst,Xtr(S,:));
    ASY = A_S' * Y;
    fprintf('done!\n')
    
    NitMax = 40;
    Tol = 1e-4;

%% Stats
test_error = @(beta) nnz(sign(Atest*beta) - Y_test)*100/ length(Y_test);
relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
summary = @(beta) [relres(beta) test_error(beta)];

%% Run SP and FALKON
fprintf('Running KRILL PCG..')
[~,statsSP] = approximate_krr(A_S,A_SS,mu,Y,summary,NitMax,Tol,'sp');
fprintf('done!\n')
fprintf('Running FALKON PCG..')
[~,statsFalk] = approximate_krr(A_S,A_SS,mu,Y,summary,NitMax,Tol,'falkon');
fprintf('done!\n')
fprintf('Running CG..')
[~,statsCG] = approximate_krr(A_S,A_SS,mu,Y,summary,NitMax,Tol,'');
fprintf('done!\n')
%% Plots
close all
loadColors
loadFont

f1 = figure(1);
semilogy(1:length(statsCG(:,1)),statsCG(:,1),'LineStyle',':','LineWidth',3,'Color',color5)
hold on
semilogy(1:length(statsFalk(:,1)),statsFalk(:,1),'LineStyle','-.','LineWidth',3,'Color',color1)
hold on 
plot(1:length(statsSP(:,1)),statsSP(:,1),'LineWidth',3,'Color',color3)
xlabel('Iteration'); ylabel('Relative residual')
legend({'No preconditioner','FALKON','KRILL (Ours)',},'FontSize',20,...
    'Location','southeast')
set(gca,'FontSize',20)

f2 = figure(2);
plot(1:length(statsCG(:,2)),statsCG(:,2),'LineStyle',':','LineWidth',3,'Color',color5)
hold on
plot(1:length(statsFalk(:,2)),statsFalk(:,2),'LineStyle','-.','LineWidth',3,'Color',color1)
hold on
plot(1:length(statsSP(:,2)),statsSP(:,2),'LineWidth',3,'Color',color3)
xlabel('Iteration'); ylabel('Test error $(\%)$')
set(gca,'FontSize',20)

%% Save
resultsPath = createFolderForExecution("susy_full10k");
saveas(f1, fullfile(resultsPath, 'susy_full10k_res.fig'))
saveas(f1, fullfile(resultsPath, 'susy_full10k_res.png'))
saveas(f2, fullfile(resultsPath, 'susy_full10k_test_err.fig'))
saveas(f2, fullfile(resultsPath, 'susy1_full10k_test_err.png'))
save(fullfile(resultsPath, 'susyfull10k_test.mat'),'statsSP','statsFalk','statsCG')
