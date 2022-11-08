function [F,AS,S] = choleskybasei(A,d,pivotselect,k,B,tol)
% RPCHOLESKYI An implicit version of RPCholesky which takes as input a
% *function* A which takes as input an index set and outputs the columns of
% A corresponding to that set. Second input specifies the diagonal.

N = length(d);
F = zeros(N,k); AS = zeros(N,k); S = zeros(k,1); i = 0;
while i < k
    s = pivotselect(d,min(B,k-i));
    S(i+1:i+length(s)) = s; 
    AS = A(s);
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

