function [Xtr, Ytr, Xts, Yts] = subsample(Xtr, Ytr, Xts, Yts, Ntr, Nts)
%SUBSAMPLE Subsamples train and test datasets.
%   Subsamples the train (Xtr, Ytr) and test (Xts, Yts) datasets using to
%   Ntr and Nts points, respectively.
n = length(Ytr);
nts = length(Yts);
idx =randperm(n,n);
idx_ts =randperm(nts,nts);
Xtr = Xtr(idx,:);
Ytr = Ytr(idx);
Xts = Xts(idx_ts,:);
Yts = Yts(idx_ts);
Xtr = Xtr(1:min(Ntr, end),:); Ytr = Ytr(1:min(Ntr,end));
Xts = Xts(1:min(Nts, end),:); Yts = Yts(1:min(Nts,end));
end
