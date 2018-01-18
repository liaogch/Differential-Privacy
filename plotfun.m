lambda = 2.25;
beta = 0.88;
%From "advanced in prospect theory cumulative representation of uncertainty"

n = 10000;
u = 0.65;

N = 10000;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

k = 1.109;
%g = 0.271;
g=(3*k-1)/n_target;

u = 0.457;
h = 0.039;




c = 1;

theta = 0.05;



W_max = 1;

beta1 = 1; 
M = c * lambda /n*power(1/n,beta1);
temp = 0; 
for i = 1:n
    temp = temp + power(i,beta1);
end
 
M = M*temp;

C = k*N/2/g/power(log(theta),2);
A = M/W_max;

A_exp = c/2/W_max;

epsilon_max = 0.001:0.001:0.02;

deri1 = 1 - A*C*beta1*epsilon_max.^(beta1 + 2) - A*epsilon_max.^beta1 + C*A*A*beta1*epsilon_max.^(2*beta1 + 2) - beta1*A*epsilon_max.^beta1;
deri11 = 1 - 2*A_exp*epsilon_max - A_exp*C*epsilon_max.^3 - A_exp^2*C*epsilon_max.^4;
%deri2 = 2*log(theta)*log(theta)./epsilon_max.^3./power(1-A*epsilon_max,3).*deri1/N^2;

%{
beta1 = 0.88;
y2 = 1 - A*C*beta1*x.^(beta1 + 2) - A*x.^beta1 + C*A*A*beta1*x.^(2*beta1 + 2) - beta1*A*x.^beta1;

beta1 = 0.44;
y3 = 1 - A*C*beta1*x.^(beta1 + 2) - A*x.^beta1 + C*A*A*beta1*x.^(2*beta1 + 2) - beta1*A*x.^beta1;

beta1 = 0.22;
y4 = 1 - A*C*beta1*x.^(beta1 + 2) - A*x.^beta1 + C*A*A*beta1*x.^(2*beta1 + 2) - beta1*A*x.^beta1;

plot(x,y1,'b',x,y2,'k',x,y3,'r',x,y4,'y')
%}
plot(epsilon_max,deri1,'b',epsilon_max,deri11,'r')

