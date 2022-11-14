%% Initialize data
load('../data/preprocessed/HIGGS.mat')
X = double(Xtr);  X_test = double(Xts);
Y = double(Ytr)'; Y_test = double(Yts)';
N = size(X,1);

% Standardization
X_mean = mean(X); X_std = std(X);
bad_idx = find(std(X) == 0);
X(:,bad_idx) = []; X_test(:,bad_idx) = [];
X_mean(bad_idx) = []; X_std(bad_idx) = [];
X = (X - X_mean) ./ X_std;
X_test = (X_test - X_mean) ./ X_std;

% Hyperparameters
mu = N*1.0e-8;
a = 20;

% Kernel
kernel = @(X1,X2) exp(-pdist2(X1,X2,"euclidean").^2/(2*a));