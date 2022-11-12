%% Load data

load_higgs

%% Test
ks = round(logspace(1,4,20));
direct_times = [];
approx_times = [];
for k = ks
    fprintf('%d\n',k)
    S = randsample(N,k,false);
    A_S = kernel(X,X(S,:));
    A_SS = A_S(S,:);

    tic; approximate_krr(A_S,A_SS,mu,Y,[],100,1e-5);
    approx_times(end+1) = toc;
    tic; w = (A_S'*A_S + mu*A_SS) \ (A_S'*Y);
    direct_times(end+1) = toc;
end

%% Figure
close all
figure(1)
loglog(ks,approx_times,'LineWidth',3)
hold on
loglog(ks,direct_times,'--','LineWidth',3)
axis([1e2 1e4 -Inf Inf])
xlabel('Number of Centers $k$')
ylabel('Computation Time (sec)')
legend({'Sketch and Precondition','Direct'},'FontSize',20,...
    'Location','southeast')
set(gca,'FontSize',20)

%% Save
saveas(gcf,'../figs/lines_crossing.png')
saveas(gcf,'../figs/lines_crossing.fig')
save('../backups/lines_crossing.mat','approx_times','direct_times')