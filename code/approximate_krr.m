function [w,stats] = approximate_krr(A_S,A_SS,mu,y,varargin)
%APPROXIMATE_KRR Sketch and precondition for approximate KRR

if ~isempty(varargin)
    summary = varargin{1};
else
    summary = @(x) [];
end

if length(varargin) > 1
    numiters = varargin{2};
else
    numiters = 100;
end

if length(varargin) > 2
    tol = varargin{3};
else
    tol = 0;
end

N = size(A_S,1); k = size(A_S,2); d = 2*k;
C = chol(A_SS + trace(A_SS)*eps*eye(size(A_SS,1)));

Phi = sparse_sign(d,N,8); PhiA_S = Phi * A_S;
B = [PhiA_S;sqrt(mu)*C];
[Q,R] = qr(B,'econ');

matvec = @(x) A_S' * (A_S * x) + mu * A_SS * x;
[w,~,stats] = mycg(matvec,@(x) R \ (R'\x),A_S'*y,tol,numiters,summary,...
    R \ (Q(1:d,:)'*(Phi*y)));
end
