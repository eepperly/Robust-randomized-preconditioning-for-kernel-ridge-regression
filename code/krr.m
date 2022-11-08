function [x,stats] = krr(A,mu,b,k,varargin)
% KRR Nystrom preconditioning for KRR
% Can be run with an explicit A or an implicit A, which is passed as a
% function which outputs the requested columns of the matrix. Implicit A
% assumes that the diagonal of the kernel is all ones unless the diagonal
% is passed in as optional argument.

if ~isempty(varargin) && ~isempty(varargin{1})
    d = varargin{1};
else
    d = ones(size(b,1),1);
end

if length(varargin) > 1 && ~isempty(varargin{2})
    summary = varargin{2};
else
    summary = @(x) [];
end

if length(varargin) > 2
    precname = varargin{3};
else 
    precname = 'nysrpc';
end

Anum = isfloat(A);
if Anum
    matvec = @(x) A*x + mu*x;
else
    matvec = @(x) kernmul(A,x) + mu*x;
end

if contains(precname, 'nys')
    if contains(precname, 'rpc')
        if ~Anum
            F = rpcholeskyi(A,d,k,min(100,ceil(k/10)));
        else
            F = rpcholesky(A,k,min(100,ceil(k/10)));
        end
    elseif contains(precname,'greedy') && ~Anum
        F = greedyi(A,d,k,1);
    elseif contains(precname,'uni') && ~Anum
        F = uniformi(A,k);
    else
        error('Other Nystrom preconditioners not yet implemented')
    end
    [U,S,~] = svd(F,'econ');
    d = 1 ./ (diag(S) .^2 + mu) - 1/mu; % Form preconditioner
    prec = @(x) U*(d.*(U'*x)) + x/mu;
else
    prec = @(x) x;
end

[x,~,stats] = mycg(matvec, prec, b,0,100,summary);
end

