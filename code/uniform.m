function [F,AS,S] = uniform(A,k,varargin)
%UNIFORM Uniform sampling for Nystrom approximation
if ~isempty(varargin)
    N = varargin{1};
elseif ~isfloat(A)
    error('Need to input size if A is an implicit matrix')
end

S = unique(randsample(N,k,false));
if isfloat(A)
    AS = A(:,S);
else
    AS = A(S);
end
A_SS = AS(S,:);
F = AS / chol(A_SS + eps*trace(A_SS)*eye(k));
end