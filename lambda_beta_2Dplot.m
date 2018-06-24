

epsilon_ref = 0;
W = 0.25;
c=1;
epsilon_max = 0.0114;
m = 100;
alpha = 0.5;

%beta = 0.01:0.01:0.8;

%lambda = zeros(1,length(beta));


%{
for i = 1:length(beta)
    lambda_search = 0.01:0.01:20;
    for j = 1:length(lambda_search)
        if check_participation( lambda_search(j),beta(i),alpha,epsilon_ref,W,c,epsilon_max,m) == 0
            lambda(i) = lambda_search(j);
            break;
        end
    end
end

plot(beta,lambda,'k','LineWidth',3);
axis([0.01 0.8 0.01 16]);

fill([beta,fliplr(beta)],[lambda,16*ones(1,length(beta))],'c');
hold on;
fill([beta,fliplr(beta)],[zeros(1,length(beta)), fliplr(lambda)],'y');


axis([0.01 0.8 0.001 16]);
xlabel('\beta');
ylabel('\lambda');
title('\epsilon^* = 0.0114');
%}


epsilon = 0.010:0.005:0.027;
for k = 1:length(epsilon)
    %lambda = 0.1./power(epsilon(i),beta);
    
    beta = 0.2:0.01:0.7;
    lambda = zeros(1,length(beta));
    for i = 1:length(beta)
    lambda_search = 0.01:0.01:20;
        for j = 1:length(lambda_search)
            if check_participation( lambda_search(j),beta(i),alpha,epsilon_ref,W,c,epsilon(k),m) == 0
                lambda(i) = lambda_search(j);
                break;
            end
        end
    end
    plot(beta,lambda,'LineWidth',3);
    hold on;
end
legend('\epsilon = 0.010','\epsilon = 0.015','\epsilon = 0.020','\epsilon = 0.025');

xlabel('\beta');
ylabel('\lambda');


%}
%{
beta_1 = 0.05:0.05:0.95;
%lambda_1 = 0.1./power(0.0086,beta_1);
lambda_1 = 0.5:0.25:11.75;

x1 = [];
y1 = [];

for i = 1:length(beta_1)
    temp = lambda_1(lambda_1 < 0.1./power(0.0086,beta_1(i)));
    len = length(temp);
    x1 = [x1 beta_1(i)*ones(1,len)];
    y1 = [y1 temp];
end

hold on;
scatter(x1,y1,'r*');

x2 = [];
y2 = [];

for i = 1:length(beta_1)
    temp = lambda_1(lambda_1 > 0.5+0.1./power(0.0086,beta_1(i)));
    len = length(temp);
    x2 = [x2 beta_1(i)*ones(1,len)];
    y2 = [y2 temp];
end

hold on;
scatter(x2,y2,'bx');
%}