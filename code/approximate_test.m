%% Set options
N = 1e4;
k = 2000;

load_chemistry_data

%% Kernel
S = randsample(N,k,false);
A_S = kernel(X,X(S,:));
A_SS = A_S(S,:);
Atest = kernel(X_test,X(S,:));
ASY = A_S' * Y;

%% Stats
test_accuracy = @(beta) norm(Atest*beta - Y_test,1) / length(Y_test);
relres = @(beta) norm(A_S'*(A_S*beta) + mu*A_SS*beta - ASY) / norm(ASY);
summary = @(beta) [relres(beta) test_accuracy(beta)];

%% Run with RPCholesky and without
[~,stats] = approximate_krr(A_S,A_SS,mu,Y,summary,100,1e-5);

%% Plots
close all

f1 = figure(1);
semilogy(stats(:,1))
xlabel('Iteration'); ylabel('Relative Residual')

f2 = figure(2);
plot(stats(:,2))
xlabel('Iteration'); ylabel('Mean Average Test Error (eV)')

%% Save
saveas(f1,'../figs/approximate_test_res.fig')
saveas(f1,'../figs/approximate_test_res.png')
saveas(f2,'../figs/approximate_test_err.fig')
saveas(f2,'../figs/approximate_test_err.png')
save('../backups/approximate_test.mat','stats')
