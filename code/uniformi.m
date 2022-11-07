function [F,AS,S] = uniformi(A,k)
%UNIFORMI An implicit version of uniform sampling
S = unique(randsample(N,k,false));
AS = A(S);
A_SS = AS(S,:);
F = AS / chol(A_SS + eps*trace(A_SS)*eye(k));
end