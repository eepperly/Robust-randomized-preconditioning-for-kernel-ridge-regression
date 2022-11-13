function [F,AS,S,nu] = uniform(A,k,varargin)
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
nu = eps(norm(AS,'fro')); %Compute shift
Y = AS + nu*sparse(S,1:k,ones(k,1),size(AS,1),size(AS,2));
A_SS = Y(S,:);
F = Y / chol(A_SS);
end