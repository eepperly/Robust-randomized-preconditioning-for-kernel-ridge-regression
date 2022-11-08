function [F,AS,S] = choleskybase(A,d,pivotselect,k,B,tol)
% CHOLESKYBASE An interface for Cholesky-based low-rank approximation

if isfloat(A)
    Afun = @(S) A(:,S);
    d = diag(A);
else
    Afun = A;
end

N = length(d);
F = zeros(N,k); AS = zeros(N,k); S = zeros(k,1); i = 0;
while i < k
    s = pivotselect(d,min(B,k-i));
    S(i+1:i+length(s)) = s; 
    AS = Afun(s);
    G = AS - F(:,1:i) * F(s,1:i)';
    R = chol(G(s,:));
    F(:,i+1:i+length(s)) = G / R;
    d = max(d - vecnorm(F(:,i+1:i+length(s)),2,2) .^ 2,0);
    i = i+length(s);
    if sum(d) < tol
        F = F(:,1:i);
        S = S(1:i);
        AS = AS(:,1:i);
        break
    end
end
end

