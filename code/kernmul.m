function b = kernmul(A,x)
%KERNMUL Multiply x by an implicit kernel matrix A
b = zeros(size(A([]),1),size(x,2)); N = size(x,1); j = 0;
while j < N
    m = min(1000,N-j);
    idx = (j+1):(j+m);
    b = b + A(idx) * x(idx,:);
    j = j + m;
end
end