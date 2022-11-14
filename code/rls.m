function F = rls(A,k,varargin)
%RLS Ridge leverage score sampling for Nystrom approximation
addpath("../recursive-nystrom")

if ~isempty(varargin)
    N = varargin{1};
elseif ~isfloat(A)
    error('Need to input size if A is an implicit matrix')
end

if length(varargin) > 1
    d = varargin{2};
elseif ~isfloat(A)
    error('Need to input size if A is an implicit matrix')
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
