function [ valuation ] = Valuation_Fun_1( x,beta,lambda,alpha,ref)
%VALUATION_FUN 此处显示有关此函数的摘要
%   此处显示详细说
if nargin == 3
    valuation = lambda * power(x,beta);
end
if nargin == 5
    if x < ref
        valuation = -lambda * power(ref-x,beta);
    else
        valuation = power(x-ref,alpha);
    end
end


