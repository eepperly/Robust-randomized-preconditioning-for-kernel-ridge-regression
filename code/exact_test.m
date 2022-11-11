%% Set options
implicit = false;
N = 1e4;
k = min(round(N/10),1000);

load_chemistry_data

%% Kernel matrix
if implicit
    d = ones(N,1);
    A = @(S) kernel(X,X(S,:));
    Atest = @(S) kernel(X_test,X(S,:));
else
    A = kernel(X,X);
    Atest = kernel(X_test,X);
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
[~,rpcholesky] = krr(A,mu,Y,k,[],summary,'rpcnys',500);
[~,greedy] = krr(A,mu,Y,k,[],summary,'greedynys',500);
[~,unif] = krr(A,mu,Y,k,[],summary,'uninys',500);
[~,rlscores] = krr(A,mu,Y,k,[],summary,'rlsnys',500);
[~,noprec] = krr(A,mu,Y,[],[],summary,'',500);

%% Plots
close all

f1 = figure(1);
semilogy(rpcholesky(:,1))
hold on
semilogy(greedy(:,1))
semilogy(unif(:,1))
semilogy(rlscores(:,1))
semilogy(noprec(:,1))
xlabel('Iteration'); ylabel('Relative Residual')
legend({'RPCholesky','Greedy','Uniform','RLS','No Preconditioner'})

f2 = figure(2);
semilogy(rpcholesky(:,2))
hold on
semilogy(greedy(:,2))
semilogy(unif(:,2))
semilogy(rlscores(:,2))
semilogy(noprec(:,2))
xlabel('Iteration'); ylabel('Test Error')
legend({'RPCholesky','Greedy','Uniform','RLS','No Preconditioner'})

%% Save
saveas(f1,'../figs/exact_test_res.fig')
saveas(f1,'../figs/exact_test_res.png')
saveas(f2,'../figs/exact_test_err.fig')
saveas(f2,'../figs/exact_test_err.png')
save('../backups/exact_test.mat','rpcholesky','unif','noprec','greedy')