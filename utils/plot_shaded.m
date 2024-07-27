function plot_shaded(x,L,U,color,varargin)
if size(x,1) == 1
    x = x';
end
if size(L,1) == 1
    L = L';
end
if size(U,1) == 1
    U = U';
end
if isstring(color) || ischar(color)
    color = hex2rgb(char(color));
end
fill([x; flipud(x)], [L; flipud(U)],color,'FaceAlpha',0.15,'EdgeColor',...
    "none");
end