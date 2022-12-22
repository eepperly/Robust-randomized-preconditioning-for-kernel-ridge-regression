function [Xtr, Xts] = standarize(Xtr, Xts)
%STANDARIZE Standarizes train and test datasets.
%   Standarizes the train Xtr and test Xts datasets using the means and
%   variances of Xtr. 
X_mean = mean(Xtr); X_std = std(Xtr);
bad_idx = find(std(Xtr) == 0);
Xtr(:,bad_idx) = []; 
% TODO: There is a bug in the way LIBSVM loads their data, which only
% affects a9a. This conditional is a hack to solve the issue. Implement
% a better fix. 
if max(bad_idx) <= size(Xts, 2)
    Xts(:,bad_idx) = [];
end
X_mean(bad_idx) = []; X_std(bad_idx) = [];
Xtr = (Xtr - X_mean) ./ X_std;
Xts = (Xts - X_mean) ./ X_std;
end