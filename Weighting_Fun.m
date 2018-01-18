function [ weighting ] = Weighting_Fun( p, u )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
weighting = exp(-power(-log(p),u));

end

