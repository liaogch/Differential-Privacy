function [ solu3 ] = Approximation_fun(  Lambda,Num,C,W1,W2 )
lambda = Lambda;
%From "advanced in prospect theory cumulative representation of uncertainty"


N = Num;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

k = 0.989;
%g = 0.271;
l=(3*k-1)/n_target;

n = 1000;
u = 0.65;



c =C;

%theta = 0.05;

Wm = W1;
Wl = W2;

W_max = 1;

beta1 = 1; 
M = c * lambda /n*power(1/n,beta1);
temp = 0; 
for i = 1:n
    temp = temp + power(i,beta1);
end
 
M = M*temp;

endpoint = W_max/M;

%C = k*N/2/l/power(log(theta),2);
C = k*N/4/l;
B = M/W_max;
i = 1;
%{
epsilon_max = 0:0.001:0.04;
deri1 = zeros(1,length(epsilon_max));
deri2 = zeros(1,length(epsilon_max));

for i = 1:length(epsilon_max)
    deri1(i) = 1 - B*C*beta1*epsilon_max(i)^(beta1 + 2) - B*epsilon_max(i)^beta1 + C*B*B*beta1*epsilon_max(i)^(2*beta1 + 2) - beta1*B*epsilon_max(i)^beta1;
    deri2(i) = 2*log(theta)*log(theta)/epsilon_max(i)^3/power(1-B*epsilon_max(i),3)/N^2;
end

figure(1)
plot(epsilon_max,deri2);

X = power(36*B^4*C+9*B^2*C^2+sqrt(3)*sqrt(432*B^8*C^2+184*B^6*C^3+27*B^4*C^4),1/3);
Y = 2^(1/3) * 3^(2/3) * B^2 * C;
Z = 2 * (2/3)^(1/3);  

s1 = sqrt(1/4/B/B + Z/X + X/Y);
s2 = sqrt(1/2/B/B - Z/X - X/Y-(1/B^3+16/B/C)/4/s1); 
s3 = sqrt(1/2/B/B - Z/X - X/Y+(1/B^3+16/B/C)/4/s1); 

solu1 = 1/4/B-0.5*s1-0.5*s2
solu2 = 1/4/B-0.5*s1+0.5*s2
solu3 = 1/4/B+0.5*s1-0.5*s3
solu4 = 1/4/B+0.5*s1+0.5*s3

opt_eps_max_appro = [opt_eps_max_appro solu3];

figure(2)
x = 0:0.001:0.05;
y = 1 - B*C*x.^3 - B*x + C*B^2*x.^4 - B*x;
plot(x,y)

1 - B*C*solu1^3 - B*solu1 + C*B^2*solu1^4 - B*solu1
1 - B*C*solu2^3 - B*solu2 + C*B^2*solu2^4 - B*solu2
1 - B*C*solu3^3 - B*solu3 + C*B^2*solu3^4 - B*solu3
1 - B*C*solu4^3 - B*solu4 + C*B^2*solu4^4 - B*solu4
%}



A1 = (-Wm-Wl)^2/4/M/M;
A2 = 4*2^(1/3)*Wl*(Wl + Wm);
A3 = power((432*C*M^4*Wl^2 + 54*C^2*M^2*Wl*(Wl + Wm)^3 +  sqrt(-6912*C^3*M^6*Wl^3*(Wl + Wm)^3 + (432*C*M^4*Wl^2+54*C^2*M^2*Wl*(Wl + Wm)^3)^2)),1/3);
A4 = 3*2^(1/3)*C*M^2;
A5 = (32*Wl)/(C*M) - (-Wm - Wl)^3/(M^3);

r1 = sqrt(A1+A2/A3+A3/A4);
r2 = sqrt(2*A1-A2/A3-A3/A4-A5/4/r1);
r3 = sqrt(2*A1-A2/A3-A3/A4+A5/4/r1);

solu1 = -(-Wm - Wl)/(4*M)-0.5*r1-0.5*r2;
solu2 = -(-Wm - Wl)/(4*M)-0.5*r1+0.5*r2;
solu3 = -(-Wm - Wl)/(4*M)+0.5*r1-0.5*r3;
solu4 = -(-Wm - Wl)/(4*M)+0.5*r1+0.5*r3;

(1/2+Wm/2/Wl-M*solu1/2/Wl)*(1-C*M*solu1^3/2/Wl)-M*solu1/2/Wl;
(1/2+Wm/2/Wl-M*solu2/2/Wl)*(1-C*M*solu2^3/2/Wl)-M*solu2/2/Wl;
(1/2+Wm/2/Wl-M*solu3/2/Wl)*(1-C*M*solu3^3/2/Wl)-M*solu3/2/Wl;
(1/2+Wm/2/Wl-M*solu4/2/Wl)*(1-C*M*solu4^3/2/Wl)-M*solu4/2/Wl;

%x = 0:0.001:endpoint;

%y = (1/2+Wm/2/Wl-M.*x/2/Wl).*(1-C*M.*x.^3/2/Wl)-M*x/2/Wl;
%plot(x,y)
%opt_eps_max_appro = [opt_eps_max_appro solu3];



end

