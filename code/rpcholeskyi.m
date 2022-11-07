function [F,AS,S] = rpcholeskyi(A,d,k,B,varargin)
% RPCHOLESKYI An implicit version of RPCholesky which takes as input a
% *function* A which takes as input an index set and outputs the columns of
% A corresponding to that set. Second input specifies the diagonal.

if ~isempty(varargin)
    tol = varargin{1};
else
    tol = 0;
end

N = length(d);
F = zeros(N,k); AS = zeros(N,k); S = zeros(k,1); i = 0;
while i < k
    i
    s = unique(randsample(N,min(B,k-i),true,d));
    S(i+1:i+length(s)) = s; 
    AS = A(s);
    G = AS - F(:,1:i) * F(s,1:i)';
    R = chol(G(s,:));
    F(:,i+1:i+length(s)) = G / R;
    d = max(d - vecnorm(F(:,i+1:i+length(s)),2,2) .^ 2,0);
    i = i+length(s);
    sum(d)
    if sum(d) < tol
        F = F(:,1:i);
        S = S(1:i);
        AS = AS(:,1:i);
        break
    end
end
end

