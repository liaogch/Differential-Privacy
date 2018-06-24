%set_env
%set path environment so that MATLAB can find the needed .m files
%without changing current working location

parent='D:\liao\MATLAB\Differential Privacy\';
path(path, parent);
path(path, [parent 'figure']);
path(path, [parent 'data']);
