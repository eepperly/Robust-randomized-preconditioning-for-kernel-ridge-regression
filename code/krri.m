function [x,stats] = krri(A,mu,b,k,varargin)
% KRRI Nystrom preconditioning for KRR with an implicitly defined kernel
% matrix. Assumes that the diagonal of the kernel is all ones unless the
% diagonal is passed in as optional argument.

if ~isempty(varargin) && ~isempty(varargin{1})
    d = varargin{1};
else
    d = ones(size(b,1));
end

if length(varargin) > 1
    summary = varargin{2};
else
    summary = @(x) [];
end

if length(varargin) > 2
    precname = varargin{3};
else 
    precname = 'nysrpc';
end

if contains(precname, 'nys')
    if contains(precname, 'rpc')
        F = rpcholeskyi(A,d,k,min(100,ceil(k/10))); % Sample
    elseif contains(precname,'greedy')
        F = greedyi(A,d,k,1); % Sample
    elseif contains(precname,'uni')
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

[x,stats] = mycg(@(x) kernmul(A,x) + mu*x, prec,...
                 b,0,100,summary);
end

