function [F,nu] = gauss_nystrom(A,k)
N = size(A,1);
Omega = orth(randn(N,k)); %Generate test matrix
if isfloat(A)
    Y = A*Omega; % Compute sketch
else
    Y = kermul(A,Omega);
end
nu = sqrt(N)*eps(norm(Y,2)); %Compute shift
Y = Y+nu*Omega;
B = Omega'*Y;
C = chol(B);
F = Y/C; %Ahat = FF'
end