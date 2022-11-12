function [F,nu] = gauss_nystrom(A,k)
[n,~] = size(A,1);
Omega = orth(randn(n,k)); %Generate test matrix
Y = A*Omega; % Compute sketch
nu = sqrt(n)*eps(norm(Y,2)); %Compute shift
Y = Y+nu*Omega;
B = Omega'*Y;
C = chol(B);
F = Y/C; %Ahat = FF'
end