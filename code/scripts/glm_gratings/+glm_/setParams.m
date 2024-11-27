function [par] = setParams(par,name,window,varargin)
%%

l = length(par);


par(l+1).align = varargin{1};
par(l+1).type = varargin{2};
par(l+1).name = name;
par(l+1).window = window;

end


