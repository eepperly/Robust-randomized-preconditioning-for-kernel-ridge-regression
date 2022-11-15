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

if length(varargin) > 3
    precname = varargin{4};
else
    precname = 'spqr';
end

if length(varargin) > 4
    verbose = varargin{5};
else
    verbose = false;
end

N = size(A_S,1); k = size(A_S,2);

if contains(precname, 'sp')
    d = 2*k;
    Phi = sparse_sign(d,N,8);
    PhiA_S = Phi * A_S;

    if contains(precname, 'qr')
        C = chol(A_SS + trace(A_SS)*eps*eye(size(A_SS,1)));
        B = [PhiA_S;sqrt(mu)*C];
        [Q,R] = qr(B,0);
        w0 = R \ (Q(1:d,:)' * (Phi*y));
    else
        H = PhiA_S'*PhiA_S + A_SS;
        R = chol(H + trace(H)*eps*eye(k));
        w0 = zeros(k,1);
    end
    prec = @(x) R \ (R'\x);
elseif contains(precname,'falkon')
    T = chol(A_SS + trace(A_SS)*eps*eye(k));
    R = chol(N/k * (T*T') + mu*eye(k));
    prec = @(x) T \ (R \ (R' \ (T' \ x)));
    w0 = zeros(k,1);
else
    error('"%s" preconditioner not implemented',precname)
end

matvec = @(x) A_S' * (A_S * x) + mu * A_SS * x;
ASy = A_S'*y;
[w,~,stats] = mycg(matvec,prec,ASy,tol,numiters,summary,w0,verbose);
end
