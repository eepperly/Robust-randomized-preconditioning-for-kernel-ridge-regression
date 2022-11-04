function [F,AS,S] = rpcholesky(A,k,B)
N = size(A,1);
F = zeros(N,k); AS = zeros(N,k); S = zeros(k,1); d = diag(A); i = 0;
while i < k
    s = unique(randsample(N,min(B,k-i),true,d));
    S(i+1:i+length(s)) = s; 
    AS = A(:,s);
    G = AS - F(:,1:i) * F(s,1:i)';
    R = chol(G(s,:));
    F(:,i+1:i+length(s)) = G / R;
    d = max(d - vecnorm(F(:,i+1:i+length(s)),2,2) .^ 2,0);
    i = i+length(s);
end
end

