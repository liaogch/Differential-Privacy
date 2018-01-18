lambda = 2.25;
beta = 1;
%From "advanced in prospect theory cumulative representation of uncertainty"


N = 2000;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

k = 0.909;
%g = 0.271;
l=(3*k-1)/n_target;

n = 1000;
u = 0.65;



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

%C = -k/l/log(theta);
B = M/W_max;

%solu = (sqrt(1+C/A)-1)/C

A1 = -18 * B^4 * k^2 * N^2 - 72 *B^5 *k *l *N^2 + 2 * B^3 * k^3 * N^3 - 12 *B^4 *k^2 *l *N^3 - 30 *B^5 *k *l^2 *N^3 - 16 * B^6 *l^3 *N^3;
A2 = 3 * B^2 * k * N * (-4 * B - 3 * B * l * N) - (-B *k *N + 2 *B^2 *l *N)^2;
A3 = (-B * k * N + 2 * B^2 *l *N)/(3 *B^2 *k *N);

solu1 = - A3 - power(2,1/3)*A2/(3*B^2*k*N*(A1+sqrt(A1^2+4*A2^3))^(1/3))+ 1/3/power(2,1/3)/B^2/k/N*(A1+sqrt(A1^2+4*A2^3))^(1/3)
solu2 = - A3 +(1+j*sqrt(3))*A2/(3*2^(2/3)*B^2*k*N*(A1+sqrt(A1^2+4*A2^3))^(1/3)) - 1/6/power(2,1/3)/B^2/k/N*(1-j*sqrt(3))*(A1+sqrt(A1^2+4*A2^3))^(1/3)
solu3 = - A3 +(1-j*sqrt(3))*A2/(3*2^(2/3)*B^2*k*N*(A1+sqrt(A1^2+4*A2^3))^(1/3)) - 1/6/power(2,1/3)/B^2/k/N*(1+j*sqrt(3))*(A1+sqrt(A1^2+4*A2^3))^(1/3)

%opt_eps_max_appro = [opt_eps_max_appro solu3];

x = 0:0.01:1;
y = (1-2*B*x).*(2+l*N*(1-B*x))-k*N*B*x.^2.*(1-B*x);
plot(x,y)

s1 = (1-2*B*solu1)*(2+l*N*(1-B*solu1))-k*N*B*solu1^2*(1-B*solu1)
s2 = (1-2*B*solu2)*(2+l*N*(1-B*solu2))-k*N*B*solu2^2*(1-B*solu2)
s3 = (1-2*B*solu3)*(2+l*N*(1-B*solu3))-k*N*B*solu3^2*(1-B*solu3)

%solu4 = -l/k+sqrt(l^2/k^2+W_max*l/M/k);

%s3 = (1-2*B*solu4)*(2+l*N*(1-B*solu4))-k*N*B*solu4^2*(1-B*solu4)