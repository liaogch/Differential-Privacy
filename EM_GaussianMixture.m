function [ GM ] = EM_GaussianMixture( X,iteration,K )
%EM_GAUSSIANMIXTURE 此处显示有关此函数的摘要
%   此处显示详细说明

len = length(X);

w = ones(len,K);

mu = ones(1,K);
p = ones(1,K)/K;
sigma = ones(1,K);

for m = 1:iteration
   
    %E step:
    for i = 1:len
        for j = 1:K
            temp = 0;
            for k = 1:K
                temp = temp + p(k)* normpdf(X(i),mu(k),sigma(k));
            end
            w(i,K) = p(j)*normpdf(X(i),mu(j),sigma(j))/temp;
        end
    end
    
    %M step:
    
    for j = 1:K
        p(j) = sum(w(:,j));
        mu(j) = w(:,j)'*X/sum(w(:,j));
        sigma(j) = w(:,j)'*((X-mu(j)).*(X-mu(j)))/sum(w(:,j));
    end
    
    
end




end

