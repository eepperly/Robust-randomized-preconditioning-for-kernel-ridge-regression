%% Set up workspace
close all; clear; clc
addpath('../utils')

%% Load data
load_higgs

%% Test
ks = round(logspace(3,4,8));
direct_times = [];
qr_times = [];
chol_times = [];
falkon_times = [];
for k = ks
    fprintf('k = %d\n',k)
    S = randsample(N,k,false);
    A_S = kernel(X,X(S,:));
    A_SS = A_S(S,:);

    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5,'spqr');
    qr_times(end+1) = toc;
    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5,'spchol');
    chol_times(end+1) = toc;
    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5,'falkon');
    falkon_times(end+1) = toc;
    tic; w = (A_S'*A_S + mu*A_SS) \ (A_S'*Y);
    direct_times(end+1) = toc;
end

%% Figure
close all
figure(1)
loglog(ks,qr_times,'LineWidth',3)
hold on
loglog(ks,chol_times,':','LineWidth',3)
loglog(ks,falkon_times,':','LineWidth',3)
loglog(ks,direct_times,'--','LineWidth',3)
axis([1e3 1e4 -Inf Inf])
xlabel('Number of Centers $k$')
ylabel('Computation Time (sec)')
legend({'SP-QR','SP-Chol','FALKON','Direct'},'FontSize',20,...
    'Location','southeast')
set(gca,'FontSize',20)

%% Save
resultsPath = createFolderForExecution("lines_crossing");
saveas(gcf, fullfile(resultsPath, 'lines_crossing.fig'))
saveas(gcf, fullfile(resultsPath, 'lines_crossing.png'))
save(fullfile(resultsPath, 'state.mat'))