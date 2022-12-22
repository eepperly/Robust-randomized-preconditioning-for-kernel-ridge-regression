function [F,AS,S] = rpcholesky(A,k,B,varargin)
%RPCHOLESKYI RPCholesky sampling for low-rank approximation
% Optional arguments (set to [] for default values):
% 1. tol: iterations stop early if trace(residulal) <= tol * trace(A)
% 2. d: diagonal of matrix A. Value is only read if A is an implicit
%    matrix, in which case it *must* be specified

if ~isempty(varargin) && ~isempty(varargin{1})
    tol = varargin{1};
else
    tol = 0;
end

if length(varargin) > 1 && ~isfloat(A)
    d = varargin{2};
    N = length(d);
elseif ~isfloat(A)
    error('Must specify diagonal if A is an implicit matrix')
else
    d = [];
    N = size(A,1);
end

[F,AS,S] = choleskybase(A,d,@(dd,m) unique(randsample(N,m,true,dd)),...
    k,B,tol);
end

