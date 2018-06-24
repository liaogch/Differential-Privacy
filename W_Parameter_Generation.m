function [ Parameter_N ] = W_Parameter_Generation( specification )
%W_PARAMETER_GENERATION 此处显示有关此函数的摘要
%   此处显示详细说明
if strcmp(specification.distribution,'Homogeneity') 
    N = specification.N;
    value = specification.value;
    Parameter_N = ones(N,1)*value;
end

if strcmp(specification.distribution,'Uniform') 
    N = specification.N;
    W_min = specification.lower;
    W_max = specification.upper;
    ud = makedist('Uniform','Lower',W_min,'Upper',W_max);
    Parameter_N = random(ud,N,1);
end

if strcmp(specification.distribution,'TruncatedNormal') 
    N = specification.N;
    nd=makedist('normal','mu',specification.average,'sigma',specification.sigma);
    td=truncate(nd,specification.lower,specification.upper);
    Parameter_N = random(td,N,1);
end

