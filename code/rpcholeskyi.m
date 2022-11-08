function [F,AS,S] = rpcholeskyi(A,d,k,B,varargin)
%RPCHOLESKYI An implicit version of RPCholesky which takes as input a
% *function* A which takes as input an index set and outputs the columns of
% A corresponding to that set. Second input specifies the diagonal.

if ~isempty(varargin)
    tol = varargin{1};
else
    tol = 0;
end
N = length(d);
[F,AS,S] = choleskybasei(A,d,@(dd,m) unique(randsample(N,m,true,dd)),...
    k,B,tol);
end

