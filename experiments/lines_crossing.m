%% Set up workspace
close all; clear; clc
addpath('../utils')
addpath('../code')

%% Load data
rng('default')
load_higgs
verbose = true;

%% Test
ks = round(logspace(3,4,11));
direct_times = [];
qr_times = [];
chol_times = [];
falkon_times = [];
for k = ks
    fprintf('k = %d\n',k)
    S = randsample(N,k,false);
    A_S = kernel(X,X(S,:));
    A_SS = A_S(S,:);

    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5,'spchol',verbose);
    chol_times(end+1) = toc;
    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5,'falkon',verbose);
    falkon_times(end+1) = toc;
    tic; w = (A_S'*A_S + mu*A_SS) \ (A_S'*Y);
    direct_times(end+1) = toc;
end

%% Figure
close all
loadColors
loadFont

figure(1)
loglog(ks,direct_times,'--','LineWidth',3,'Color',color4)
hold on
loglog(ks,falkon_times,'-.','LineWidth',3,'Color',color1)
loglog(ks,chol_times,'-','LineWidth',3,'Color',color3)
axis([min(ks) max(ks) -Inf Inf])
xlabel('Number of Centers $k$')
ylabel('Computation Time (sec)')
legend({'Direct','FALKON','KRILL'},'FontSize',20,...
    'Location','southeast')
set(gca,'FontSize',20)

%% Save
resultsPath = createFolderForExecution("lines_crossing");
saveas(gcf, fullfile(resultsPath, 'lines_crossing.fig'))
saveas(gcf, fullfile(resultsPath, 'lines_crossing.png'))
save(fullfile(resultsPath, 'state.mat'),'N','ks','mu','qr_times',...
    'chol_times','falkon_times','direct_times')