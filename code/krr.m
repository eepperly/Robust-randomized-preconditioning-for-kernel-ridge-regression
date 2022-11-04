function x = krr(A,mu,b,k)
F = rpcholesky(A,k,32); % Sample
[U,S,~] = svd(F,'econ'); N = size(A,1);

% Naive
d = mu ./ (diag(S) .^2 + mu) - 1; % Form preconditioner
x = pcg(@(x) A*x + mu*x,b,1e-4,100,@(x) U*(d.*(U'*x)) + x);
end

