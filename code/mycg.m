function [x,iter,stats] = mycg(matvec,prec,b,tol,maxit,varargin)
summary = [];
if ~isempty(varargin)
    summary = varargin{1};
end

if length(varargin) > 1 && ~isempty(varargin{2})
    x = varargin{2};
    r = b - matvec(x);
else
    x = zeros(size(b)); 
    r = b;
end

if length(varargin) > 2
    verbose = varargin{3};
else
    verbose = false;
end

stats = [];
bnorm = norm(b); rnorm = bnorm;
z = prec(r); p = z;
for iter = 1:maxit
    if verbose
        fprintf('%d\t%e\n',iter,rnorm/bnorm)
    end
    v = matvec(p);
    zr = z'*r; eta = zr / (v'*p);
    x = x + eta*p;
    r = r - eta*v;
    z = prec(r);
    gamma = z'*r/zr;
    p = z + gamma*p;
    rnorm = norm(r);
    if ~isempty(summary); stats(end+1,:) = summary(x); end %#ok<AGROW> 
    if rnorm <= tol * bnorm; break; end
end
end
