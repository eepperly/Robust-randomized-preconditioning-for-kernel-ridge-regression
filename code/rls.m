function F = rls(A,k,varargin)
%RLS Ridge leverage score sampling for Nystrom approximation
% Optional arguments (set to [] for default values):
% 1. d: diagonal of matrix A. Value is only read if A is an implicit
%    matrix, in which case it *must* be specified

addpath("../recursive-nystrom")

if length(varargin) > 1 && ~isfloat(A)
    d = varargin{2};
    N = length(d);
elseif ~isfloat(A)
    error('Must specify diagonal if A is an implicit matrix')
else
    d = [];
    N = size(A,1);
end

if isfloat(A)
    kernfunc = @(X,rows,cols) Amat_to_kernel(A,rows,cols);
else
    kernfunc = @(X,rows,cols) Afun_to_kernel(A,rows,cols,d);
end

[C,W] = recursiveNystrom((1:N)',k,kernfunc);
R = chol(W + trace(W)*eps*eye(size(W,1)));
F = C * R';
end

function submatrix = Afun_to_kernel(A,rows,cols,d)
    if isempty(cols)
        submatrix = d; return
    end
    AS = A(cols);
    submatrix = AS(rows,:);
end

function submatrix = Amat_to_kernel(A,rows,cols)
    if isempty(cols)
        submatrix = diag(A); return
    end
    submatrix = A(rows,cols);
end
