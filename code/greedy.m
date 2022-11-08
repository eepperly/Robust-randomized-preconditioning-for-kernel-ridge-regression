function [F,AS,S] = greedy(A,k,B,varargin)
%GREEDY Greedy pivoting for psd low-rank approximation

if ~isempty(varargin)
    tol = varargin{1};
else
    tol = 0;
end

if length(varargin) > 1 && ~isfloat(A)
    d = varargin{2};
elseif ~isfloat(A)
    error('Must specify diagonal if A is an implicit matrix')
else
    d = [];
end

[F,AS,S] = choleskybase(A,d,@greedyselect,k,B,tol);
end

function idx = greedyselect(d,m)
    [~,idx] = maxk(d,m);
end