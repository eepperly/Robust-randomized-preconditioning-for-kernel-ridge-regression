%% Set up workspace
close all; clear; clc
addpath('../utils')
addpath('../code')
resultsPath = createFolderForExecution("lines_crossing");

%% Load data
rng('default')
load_susy
verbose = true;

%% Test
num_runs = 21;
ks = round(logspace(2,4,num_runs));
direct_times = zeros(num_runs,1);
krill_times = zeros(num_runs,1);
falkon_times = zeros(num_runs,1);
krill_iters = zeros(num_runs,1);
falkon_iters = zeros(num_runs,1);
trials = 5;
for run = 1:num_runs
    k = ks(run);
    fprintf('k = %d\n',k)
    S = randsample(N,k,false);
    A_S = kernel(X,X(S,:));
    A_SS = A_S(S,:);
    
    for trial = 1:trials
        fprintf('KRILL (k=%d,tr=%d)\n',k,trial)
        tic; 
        [~,summary] = approximate_krr(A_S,A_SS,mu,Y,@(x) 1,300,1e-5,...
            'spchol',verbose);
        krill_times(run) = krill_times(run) + toc/trials;
        krill_iters(run) = krill_iters(run) + length(summary)/trials;
        fprintf('FALKON (k=%d,tr=%d)\n',k,trial)
        tic;
        [~,summary] = approximate_krr(A_S,A_SS,mu,Y,@(x) 1,300,1e-5,...
            'falkon',verbose);
        falkon_times(run) = falkon_times(run) + toc/trials;
        falkon_iters(run) = falkon_iters(run) + length(summary)/trials;
        fprintf('Direct (k=%d,tr=%d)\n',k,trial)
        tic; w = (A_S'*A_S + mu*A_SS) \ (A_S'*Y);
        direct_times(run) = direct_times(run) + toc/trials;
    end

    save(fullfile(resultsPath, 'state.mat'),'N','ks','mu','krill_times',...
        'falkon_times','direct_times','krill_iters','falkon_iters')
end

%% Figure
close all
loadColors
loadFont

figure(1)
loglog(ks,direct_times,'--','LineWidth',3,'Color',color4)
hold on
loglog(ks,falkon_times,'-.','LineWidth',3,'Color',color1)
loglog(ks,krill_times,'-','LineWidth',3,'Color',color3)
axis([min(ks) max(ks) -Inf Inf])
xlabel('Number of centers $k$')
ylabel('Computation time (sec)')
legend({'Direct','FALKON','KRILL'},'FontSize',20,...
    'Location','southeast')
set(gca,'FontSize',20)

%% Save
saveas(gcf, fullfile(resultsPath, 'lines_crossing.fig'))
saveas(gcf, fullfile(resultsPath, 'lines_crossing.png'))
save(fullfile(resultsPath, 'state.mat'),'N','ks','mu','krill_times',...
    'falkon_times','direct_times','krill_iters','falkon_iters')