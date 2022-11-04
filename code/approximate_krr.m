function w = approximate_krr(A_S,A_SS,mu,y)
N = size(A_S,1); k = size(A_S,2); d = 2*k;
C = chol(A_SS + trace(A_SS)*eps*eye(size(A_SS,1)));

Phi = sparse_sign(d,N,8);
[Q,R] = qr([Phi*A_S;sqrt(mu)*C],'econ');

matvec = @(x) A_S' * (A_S * x) + mu * A_SS * x;
w = pcg(matvec,A_S'*y,1e-4,100,R',R,R \ (Q(1:d,:)'*(Phi*y)));
end
