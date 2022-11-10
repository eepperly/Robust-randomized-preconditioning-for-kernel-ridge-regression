function [x,iter,stats] = mycg(matvec,prec,b,tol,maxit,varargin)
summary = [];
if ~isempty(varargin)
    summary = varargin{1};
end
stats = [];
x = zeros(size(b)); r = b; bnorm = norm(b); rnorm = bnorm;
z = prec(r); p = z;
for iter = 1:maxit
    fprintf('%d\t%e\n',iter,rnorm/bnorm)
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