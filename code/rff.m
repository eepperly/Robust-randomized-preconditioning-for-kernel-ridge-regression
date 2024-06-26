function U = rff(X,mu,sigma,s)
% Constructs preconditioner for KRR with Gaussian kernel using
%random fourier features approach of ACW (2017)
[~,d] = size(X);  
W = 1/sigma*(randn(s,d)); b = unifrnd(0,2*pi,s,1);
Z = sqrt(2/s)*cos(W*X'+b)';
L = chol(Z'*Z+mu*eye(s),'lower'); 
U = L\Z';

