function [F,nu] = gauss_nystrom(A,k,varargin)
if isfloat(A)
    N = size(A,1);
else
    N = varargin{1};
end
Omega = orth(randn(N,k)); %Generate test matrix
if isfloat(A)
    Y = A*Omega; % Compute sketch
else
    Y = kernmul(A,Omega);
end
nu = sqrt(N)*eps(norm(Y,2)); %Compute shift
Y = Y+nu*Omega;
B = Omega'*Y;
C = chol(B);
F = Y/C; %Ahat = FF'
end