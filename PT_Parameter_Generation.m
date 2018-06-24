function [ Parameter_N ] = PT_Parameter_Generation( specification)
%PT_PARAMETER 此处显示有关此函数的摘要
%   此处显示详细说明
if strcmp(specification.distribution,'TruncatedGaussian') 
    ave = specification.ave;
    sigma = specification.sigma;
    lower = specification.lower;
    upper = specification.upper;
    N = specification.N;
    
    nd=makedist('normal','mu',ave,'sigma',sigma);
    td=truncate(nd,lower,upper);
    Parameter_N = random(td,N,1);
end    



if strcmp(specification.distribution,'Gamma') 
        
    shape = specification.shape;
    scale = specification.scale;
        %rate = 1/scale;
    N = specification.N;
      
        
    gd = makedist('gamma','a',shape,'b',scale);
    Parameter_N = random(gd,N,1);
end

if strcmp(specification.distribution,'Homogeneity') 
    N = specification.N;
    value = specification.value;
    Parameter_N = ones(N,1)*value;
end

if strcmp(specification.distribution,'Uniform')
   lower = specification.mean - specification.range/2;
   upper = specification.mean + specification.range/2;
   N = specification.N;
   ud = makedist('uniform','Lower',lower,'Upper',upper);
   Parameter_N = random(ud,N,1); 
end
    %{
    ave = specification.ave;
    sigma = specification.sigma;
    probability = specification.probability;
    lower = specification.lower;
    upper = specification.upper;
    N = specification.N;
    
    len = length(ave);

    mean = ave*probability';


    parameter = zeros(N,len);


    for i =1:len
        if sigma(i)==0
            parameter(:,i) = ones(N,1)*ave(i);
        else
            nd=makedist('normal','mu',ave(i),'sigma',sigma(i));
            td=truncate(nd,lower(i),upper(i));
            parameter(:,i) = random(td,N,1);
        end
    end



    Parameter_N = parameter(:,len);

    if i~= 1
        Parameter_N = parameter(:,len);
        pro = rand(N,1);
        for i = 1:N
            for j = 1:len-1
                if pro(i)< sum(probability(1:j))
                    Parameter_N(i) = parameter(i,j);
                    break;
                end
            end
        
        end
    end
    %}
