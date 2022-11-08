function [F,AS,S] = uniform(A,k)
%UNIFORM Uniform sampling for Nystrom approximation
S = unique(randsample(N,k,false));
if isfloat(A)
    AS = A(:,S);
else
    AS = A(S);
end
A_SS = AS(S,:);
F = AS / chol(A_SS + eps*trace(A_SS)*eye(k));
end