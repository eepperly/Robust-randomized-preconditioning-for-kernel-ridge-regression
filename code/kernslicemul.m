function b = kernslicemul(A_S,x,sz,varargin)
%KERNSLICEMUL Multiply vector x by implicit kernel matrix A_S, A_S', or
%A_S'*A_S
if ~isempty(varargin) 
    option = varargin{1};
else
    option = 'both';
end

if strcmp(option,'both')
    k = size(x,1);
    b = kernslicemuladj(A_S,kernslicemulfor(A_S,x,sz),k);
elseif strcmp(option,'forward')
    b = kernslicemulfor(A_S,x,sz);
elseif strcmp(option,'adjoint')
    b = kernslicemuladj(A_S,x,sz);
else
    error('Kernel slice multiplication option "%s" not recognized',...
        option);
end
end

function b = kernslicemulfor(A_S,x,N)
    b = zeros(N,size(x,2));
    k = size(x,1);
    j = 0;
    while j < N
        stride = min(k,N-j);
        b((j+1):(j+stride),:) = A_S((j+1):(j+stride)) * x;
        j = j + stride;
    end
end

function b = kernslicemuladj(A_S,x,k)
    b = zeros(k,size(x,2));
    N = size(x,1);
    j = 0;
    while j < N
        stride = min(k,N-j);
        b = b + A_S((j+1):(j+stride))' * x((j+1):(j+stride),:);
        j = j + stride;
    end
end