function [w,stats] = approximate_krr(A_S,A_SS,mu,y,varargin)
%APPROXIMATE_KRR Sketch and precondition for approximate KRR
% Optional arguments (set to [] for default values):
% 1. summary: function mapping current CG iterate to a row vector of
%    information to be returned in the 'stats' output.
% 2. numiters: maximum number of CG iterations.
% 3. tol: relative tolerance for CG, i.e, CG stops after the residual is
%    reduced to 'tol' times its inital value (defaults to 0).
% 4. precname: name of preconditioner (defaults to 'spchol', the sketch and
%    precondition approach using Cholesky factorization).
% 5. verbose: whether to print iteration information (defaults to false).

    if ~isempty(varargin) && ~isempty(varargin{1})
        summary = varargin{1};
    else
        summary = [];
    end

    if length(varargin) > 1 && ~isempty(varargin{2})
        numiters = varargin{2};
    else
        numiters = 100;
    end

    if length(varargin) > 2 && ~isempty(varargin{3})
        tol = varargin{3};
    else
        tol = 0;
    end

    if length(varargin) > 3 && (((ischar(varargin{4})...
            || isstring(varargin{4})) && strcmp(varargin{4},'')) ...
            || ~isempty(varargin{4}))
        precname = varargin{4};
    else
        precname = 'spchol';
    end

    if length(varargin) > 4  && ~isempty(varargin{5})
        verbose = varargin{5};
    else
        verbose = false;
    end
    %% Check errors

    if size(A_SS,1) ~= size(A_SS,2)
        error('A_SS must be a square matrix')
    end

    if isfloat(A_S)
        if size(A_S,1) ~= size(y,1)
            error('A_S must have as many rows as y')
        end

        if size(A_S,2) ~= size(A_SS,1)
            error('A_S must have as many columns as the dimensions of A_SS')
        end
    end

    %% Solve
    N = size(y,1); k = size(A_SS,1);

    if contains(precname, 'sp')
        assert(isfloat(A_S))
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
        assert(isfloat(A_S))
        T = chol(A_SS + trace(A_SS)*eps*eye(k));
        R = chol(N/k * (T*T') + mu*eye(k));
        prec = @(x) T \ (R \ (R' \ (T' \ x)));
        w0 = zeros(k,1);
    elseif strcmp(precname,'')
        prec = @(x) x;
        w0 = zeros(k,1);
    else
        error('"%s" preconditioner not implemented',precname)
    end

    if isfloat(A_S)
        matvec = @(x) A_S' * (A_S * x) + mu * A_SS * x;
        ASy = A_S'*y;
    else
        matvec = @(x) kernslicemul(A_S,x,N) + mu * A_SS * x;
        ASy = kernslicemul(A_S,y,k,'adjoint');
    end

    [w,~,stats] = mycg(matvec,prec,ASy,tol,numiters,summary,w0,verbose);
end
