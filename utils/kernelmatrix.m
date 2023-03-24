function K = kernelmatrix(X1, X2, kerneltype, bandwidth)
%KERNELMATRIX Computes the kernel matrix K(X1, X2).
%   Computes the kernel matrix of the two samples. Valid options for the 
% kernel type are 'gaussian' and 'laplace'. 
if contains(kerneltype, 'gaussian')
    K = exp(-pdist2(X1,X2,"euclidean").^2 / (2*bandwidth^2));
elseif contains(kerneltype, 'laplace')
    K = exp(-pdist2(X1,X2,"minkowski",1)/bandwidth);
else
    error('Kernel ' + kerneltype + ' not implemented')
end
end