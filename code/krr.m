function [x,stats] = krr(A,mu,b,k,varargin)
% KRR Nystrom preconditioning for KRR
% Can be run with an explicit A or an implicit A, which is passed as a
% function which outputs the requested columns of the matrix. Implicit A
% assumes that the diagonal of the kernel is all ones unless the diagonal
% is passed in as optional argument.
% Optional arguments (set to [] for default values):
% 1. d: diagonal of matrix A. Value is only read if A is an implicit
%    matrix. Defaults to all one's if not specified.
% 2. summary: function mapping current CG iterate to a row vector of
%    information to be returned in the 'stats' output.
% 3. precname: name of preconditioner (defaults to 'nysrpc', RPCholesky
%    preconditioning).
% 4. numiters: maximum number of CG iterations (defaults to 100).
% 5. choltol: relative tolerance for Cholesky-based Nystroms (defaults to zero).
% 6. pcgtol: relative tolerance for CG, i.e, CG stops after the residual is
%    reduced to 'pcgtol' times its inital value (defaults to 0).
% 7. verbose: whether to print iteration information (defaults to false).

if isfloat(A)
    d = diag(A);
elseif ~isempty(varargin) && ~isempty(varargin{1})
    d = varargin{1};
else
    d = ones(size(b,1),1);
end

if length(varargin) > 1 && ~isempty(varargin{2})
    summary = varargin{2};
else
    summary = @(x) [];
end

if length(varargin) > 2 &&  ~isempty(varargin{3})
    precname = varargin{3};
else
    precname = 'nysrpc';
end

if length(varargin) > 3 &&  ~isempty(varargin{4})
    numiters = varargin{4};
else
    numiters = 100;
end

if length(varargin) > 4 &&  ~isempty(varargin{5})
    choltol = varargin{5};
else
    choltol = [];
end

if length(varargin) > 5 &&  ~isempty(varargin{6})
    pcgtol = varargin{6};
else
    pcgtol = [];
end

if length(varargin) > 6 &&  ~isempty(varargin{7})
    verbose = varargin{7};
else
    verbose = false;
end

Anum = isfloat(A);
if Anum
    matvec = @(x) A*x + mu*x;
    N = size(A,1);
else
    matvec = @(x) kernmul(A,x) + mu*x;
    N = size(d,1);
end

if contains(precname, 'nys')
    if contains(precname, 'rpc')
        F = rpcholesky(A,k,min(100,ceil(k/10)),choltol,d);
    elseif contains(precname,'greedy')
        F = greedy(A,k,1,choltol,d);
    elseif contains(precname,'uni')
        [F,~,~,nu] = uniform(A,k,N);
    elseif contains(precname,'rls')
        F = rls(A,k,N,d); %#ok<CMRLS>
    elseif contains(precname,'gauss')
        [F,nu] = gauss_nystrom(A,k,N);
    else
        error('Other Nystrom preconditioners not yet implemented')
    end
    [U,S,~] = svd(F,'econ');
    if contains(precname,'gauss') || contains(precname,'uni')
        d = 1./(max(diag(S).^2-nu,0)+mu)-1/mu; %Removes shift nu
    else
        d = 1 ./ (diag(S) .^2 + mu) - 1/mu; % Form preconditioner
    end
    prec = @(x) U*(d.*(U'*x)) + x/mu;
else
    prec = @(x) x;
end

[x,~,stats] = mycg(matvec,prec,b,pcgtol,numiters,summary,[],verbose);
end

