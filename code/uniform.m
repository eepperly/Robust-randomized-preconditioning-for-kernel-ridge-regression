function [F,AS,S,nu] = uniform(A,k,varargin)
%UNIFORM Uniform sampling for Nystrom approximation
% Optional arguments (set to [] for default values):
% 1. N: size of matrix A. Value is only used if A is an implicit matrix, in
%    which case N *must* be specified

if ~isfloat(A)
    if isempty(varargin)
        error('Need to input size if A is an implicit matrix')
    end
    N = varargin{1};
    assert(mod(N,1) == 0)
else
    N = size(A,1);
end

S = unique(randsample(N,k,false));
if isfloat(A)
    AS = A(:,S);
else
    AS = A(S);
end
nu = eps(norm(AS,'fro')); %Compute shift
Y = AS + nu*sparse(S,1:k,ones(k,1),size(AS,1),size(AS,2));
A_SS = Y(S,:);
F = Y / chol(A_SS);
end