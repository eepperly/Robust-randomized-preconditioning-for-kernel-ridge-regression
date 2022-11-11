%% Initialize data
load('../data/homo.mat')
X_test = X(N+1:(2*N),:); Y_test = Y(N+1:(2*N));
X = X(1:N,:); Y = Y(1:N);

% Standardization
X_mean = mean(X); X_std = std(X);
bad_idx = find(std(X) == 0);
X(:,bad_idx) = []; X_test(:,bad_idx) = [];
X_mean(bad_idx) = []; X_std(bad_idx) = [];
X = (X - X_mean) ./ X_std;
X_test = (X_test - X_mean) ./ X_std;

% Hyperparameters
mu = N*1.0e-8;
a = 5120;

% Kernel
kernel = @(X1,X2) exp(-pdist2(X1,X2,"minkowski",1)/a);