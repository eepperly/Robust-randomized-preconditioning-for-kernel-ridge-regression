function [F,AS,S] = greedyi(A,d,k,B,varargin)
%GREEDYI An implicit version of greedy pivoting which takes as input a
% *function* A which takes as input an index set and outputs the columns of
% A corresponding to that set. Second input specifies the diagonal.

if ~isempty(varargin)
    tol = varargin{1};
else
    tol = 0;
end
[F,AS,S] = choleskybasei(A,d,@greedyselect,k,B,tol);
end

function idx = greedyselect(d,m)
    [~,idx] = maxk(d,m);
end